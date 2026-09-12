#!/usr/bin/env python
"""Generador de la trazabilidad desde la fuente canónica única
(docs/claims_registry.yaml) — la trazabilidad se GENERA, no se duplica.

Produce DOS vistas:
  - docs/matriz_trazabilidad.md          (la matriz clásica, migrada)
  - docs/ontology_nomology_observables.md (entidad → ley → observable
    → estado de evidencia, para la v36)

El test de no-divergencia (tests/test_claims_registry.py) regenera
ambas y falla si difieren de lo commiteado: un cambio de estatuto
(p. ej. #13, #15, 5E) hecho en el YAML no puede dejar dos documentos
contradictorios, y un cambio a mano en los .md sin tocar el YAML
rompe la suite.

Uso: python scripts/make_traceability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "docs" / "claims_registry.yaml"
MATRIX = ROOT / "docs" / "matriz_trazabilidad.md"
ONTOLOGY = ROOT / "docs" / "ontology_nomology_observables.md"

GENERATED_NOTE = (
    "<!-- GENERADO por scripts/make_traceability.py desde "
    "docs/claims_registry.yaml — NO editar a mano: el test de "
    "no-divergencia (tests/test_claims_registry.py) falla si este "
    "fichero difiere de la regeneración. -->")

NEGATIVE_SECTIONS = (
    ("resultado-negativo",
     "Resultados negativos (experimento EJECUTADO, desenlace nulo o "
     "contrario, publicado como tal)"),
    ("claim-no-derivado",
     "Claims no derivados (la afirmación existe; su derivación, no — "
     "distinto de un negativo: no hay test ejecutable que la decida "
     "aún)"),
    ("experimento-no-ejecutado",
     "Experimentos no ejecutados (protocolo preinscrito, datos "
     "ausentes, fallo cerrado — sin veredicto en ningún sentido)"),
)


def load_registry() -> dict:
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))


def _md_cell(text: str) -> str:
    """Celda de tabla markdown: escapa pipes y colapsa a una línea."""
    return " ".join(str(text).split()).replace("|", "\\|")


def _paths_cell(text: str) -> str:
    """Columna de rutas: cada segmento «a + b» va en backticks."""
    text = str(text).strip()
    if text == "—":
        return "—"
    return " + ".join(f"`{p.strip()}`" for p in text.split(" + "))


def render_matrix(reg: dict) -> str:
    lines = [
        "# Matriz canónica manuscrito ↔ código ↔ test ↔ estatuto",
        "",
        GENERATED_NOTE,
        "",
        "Trazabilidad explícita del programa (propuesta del análisis "
        "v36, punto 8; el Apéndice H exige que cada pieza resista una "
        "prueba objetiva y que cada frente declare qué le falta). Los "
        "estatutos siguen la v35.1: «interna demostrada» = "
        "comprobación interna de la implementación superada (E8), "
        "nunca demostración física. Las celdas «—» declaran un hueco "
        "real, no lo esconden. La fuente canónica es "
        "`docs/claims_registry.yaml` (falsadores, datasets y "
        "categorías negativas viven allí); "
        "`tests/test_traceability.py` verifica que cada ruta citada "
        "existe y `tests/test_claims_registry.py` que esta vista no "
        "diverge de la fuente.",
        "",
        "| Elemento | Manuscrito | Código | Test | Estatuto |",
        "|---|---|---|---|---|",
    ]
    for c in reg["claims"]:
        lines.append(
            f"| {_md_cell(c['entity'])} | {_md_cell(c['law'])} | "
            f"{_paths_cell(c['implementation'])} | "
            f"{_paths_cell(c['test'])} | "
            f"{_md_cell(c['evidence_level'])} |")
    return "\n".join(lines) + "\n"


def render_ontology(reg: dict) -> str:
    claims = reg["claims"]
    lines = [
        "# Ontología → nomología → observables (vista para la v36)",
        "",
        GENERATED_NOTE,
        "",
        "Cada entidad del programa con su ley, su observable (si lo "
        "tiene), el dataset que lo decide y su falsador declarado. "
        "Vocabulario de estatutos v35.1; tres categorías negativas "
        "DISTINTAS — resultado negativo ≠ claim no derivado ≠ "
        "experimento no ejecutado — porque colapsan responsabilidades "
        "distintas: lo medido y adverso, lo aún no derivable, y lo "
        "bloqueado por datos.",
        "",
        "**FRB no entra**: sin dataset con versión/DOI persistente, "
        "sin observable consumido por el MCMC y sin relación precisa "
        "con el parámetro bariónico que restringiría, es literatura "
        "relevante, no evidencia del modelo (control bariónico "
        "anclado hoy: kSZ 2604.19744/45).",
        "",
        "| Entidad | Ley | Observable | Dataset | Falsador | "
        "Estatuto | Categoría | Artefacto |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in claims:
        lines.append(
            f"| {_md_cell(c['entity'])} | {_md_cell(c['law'])} | "
            f"{_md_cell(c['observable'])} | {_md_cell(c['dataset'])} | "
            f"{_md_cell(c['falsifier'])} | "
            f"{_md_cell(c['evidence_level'])} | "
            f"`{c['status']}` | {_md_cell(c['artifact'])} |")
    for status, title in NEGATIVE_SECTIONS:
        subset = [c for c in claims if c["status"] == status]
        lines += ["", f"## {title}", ""]
        if not subset:
            lines.append("(ninguna fila en esta categoría)")
        for c in subset:
            lines.append(f"- **{_md_cell(c['entity'])}** "
                         f"({c['claim_id']}): "
                         f"{' '.join(str(c['derivation']).split())}")
    return "\n".join(lines) + "\n"


def main() -> int:
    reg = load_registry()
    MATRIX.write_text(render_matrix(reg), encoding="utf-8")
    ONTOLOGY.write_text(render_ontology(reg), encoding="utf-8")
    print(f"{len(reg['claims'])} claims → {MATRIX.name}, "
          f"{ONTOLOGY.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
