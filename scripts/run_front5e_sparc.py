#!/usr/bin/env python
"""Frente 5E — análisis de falsación cruzada Sculptor ↔ SPARC.

Flujo (preinscrito en results/2026-08-16_front5e_sparc/):

1. Carga la configuración CONGELADA (A de Sculptor; nunca se estima
   aquí — candado test_no_sparc_fit_of_A).
2. Intenta cargar el catálogo SPARC ingerido con manifest.
3. SIN datos: escribe el veredicto DATA_UNAVAILABLE preinscrito
   (determinista, sin timestamps — reproducible byte a byte) y
   termina. No fabrica fixtures, no usa mirrors, no emite veredicto
   observacional.
4. CON datos (tras la ingesta real): schema_report primero; después
   la comparación primaria M0 vs M1 por galaxia (5E-A), agregados,
   bootstrap, figuras y per_galaxy.csv — todo según la preinscripción.

Uso: python scripts/run_front5e_sparc.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamics.sculptor_transfer import frozen_config  # noqa: E402
from dynamics.sparc_data import load_sparc_catalogue  # noqa: E402

OUT = (Path(__file__).resolve().parent.parent / "results"
       / "2026-08-16_front5e_sparc")


def write_unavailable_verdict(out_dir: Path) -> dict:
    """El veredicto preinscrito para el caso sin datos — determinista
    (sin timestamps: la fecha vive en la preinscripción y en git)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = frozen_config()
    summary = {
        "front": "5E — falsación cruzada Sculptor ↔ SPARC",
        "status": "DATA_UNAVAILABLE",
        "A_frozen": cfg.A_sculptor,
        "verdict": (
            "5E observacional ABIERTO por ausencia de datos "
            "ingeridos. El resultado estructural 5C permanece con su "
            "alcance declarado y NO se eleva a veredicto "
            "observacional. La A transferida queda congelada y "
            "publicada en la preinscripción; el pipeline completo "
            "está construido y testeado con fixtures sintéticos, "
            "listo para la ingesta."),
        "blocking": "proxy de salida: astroweb.cwru.edu y "
                    "vizier.cds.unistra.fr con CONNECT 403 "
                    "(data/sparc/INGESTA_ESTADO.md)",
        "rules_in_force": [
            "checksums abiertos hasta la primera descarga real",
            "sin fixtures observacionales",
            "sin mirrors no verificados byte a byte",
            "sin veredicto SPARC observacional",
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (out_dir / "report.md").write_text(
        "# Frente 5E — estado: DATA_UNAVAILABLE\n\n"
        "La preinscripción está congelada "
        "(`preregistration.{json,md}`, generada ANTES de intentar la "
        f"ingesta): **A = {cfg.A_sculptor:.6e} (M⊙/pc³)^(−3/2)** "
        f"(Υ⋆ = {cfg.upsilon_sculptor:.0f}; sensibilidad declarada "
        f"{cfg.A_sculptor_sensitivity[0]:.3e} / "
        f"{cfg.A_sculptor_sensitivity[1]:.3e}).\n\n"
        "La fuente primaria y VizieR respondieron CONNECT 403 desde "
        "el proxy del entorno (evidencia y desbloqueo en "
        "`data/sparc/INGESTA_ESTADO.md`). Por la regla de fallo "
        "preinscrita:\n\n"
        "- el 5E observacional queda **ABIERTO por ausencia de datos "
        "ingeridos**;\n"
        "- el resultado estructural 5C permanece con su alcance "
        "declarado y **no se eleva a veredicto observacional**;\n"
        "- los checksums siguen abiertos; no se fabrican fixtures "
        "observacionales; no se usan mirrors sin prueba de identidad "
        "byte a byte.\n\n"
        "El pipeline completo (composición en cuadratura, χ², "
        "diagnósticos B_inner/F_outer/ley ×5, agregados con "
        "bootstrap, fila per-galaxy con `A_fitted = false`) está "
        "construido como funciones puras que reciben la config "
        "inmutable, y testeado con fixtures sintéticos — la ingesta "
        "real es un comando y el análisis observacional otro.\n",
        encoding="utf-8")
    return summary


def main() -> None:
    cfg = frozen_config()
    print(f"A congelada (entrada, no estimada): {cfg.A_sculptor:.6e}")
    try:
        load_sparc_catalogue()
    except FileNotFoundError as exc:
        print(str(exc))
        write_unavailable_verdict(OUT)
        print(f"Veredicto DATA_UNAVAILABLE: {OUT}/report.md")
        return
    raise SystemExit(
        "Datos presentes: ejecutar primero la inspección de esquema "
        "(python -m dynamics.sparc_data --inspect) y completar el "
        "parser según schema_report.md antes del análisis "
        "observacional (contrato del 5E).")


if __name__ == "__main__":
    main()
