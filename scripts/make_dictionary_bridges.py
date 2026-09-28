#!/usr/bin/env python
"""Genera docs/diccionario_puentes.md desde docs/claims_registry.yaml — PR-7 de la orden del 28-sep.

La tabla de puentes fundamental → cosmología (la nota del 18-sep) pasa de nota a VISTA REGENERABLE, como
la matriz de trazabilidad: cada puente (una magnitud que el tramo pre-geométrico debería entregar a la
cosmología, o una constante que la cosmología usa y que debería salir del tramo pre-geométrico) se
clasifica leyendo el `status` y el `evidence_level` de las filas del registro que lo sostienen:

    derivado      — todas las filas son «interno» (comprobación interna superada, E8) sin señal de calibración
    condicional   — alguna fila «condicional» y ninguna calibrada/ausente
    calibrado     — alguna fila con status «calibrado», o cuyo evidence_level dice «calibrad…» (número ajustado, con
                    estatuto declarado), o cuyo dataset declara un «ancla externa» (constante tomada de fuera, no ingerida)
    convencional  — la fila que lo fija declara «convención» en su estatuto (elección, no medida ni derivación)
    ausente       — alguna fila es hueco declarado, claim no derivado o experimento no ejecutado

Los campos leídos son exactamente `status`, `evidence_level` y `dataset`; los marcadores, las cadenas literales de
arriba. La clasificación es una REGLA sobre el registro, no un juicio: si una fila cambia de estatuto, la tabla
cambia sola. `tests/test_dictionary_bridges.py` regenera la vista y falla si el fichero difiere (no
divergencia) y si alguna fila citada no existe.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "docs" / "claims_registry.yaml"
OUT = ROOT / "docs" / "diccionario_puentes.md"

GENERATED_NOTE = ("<!-- GENERADO por scripts/make_dictionary_bridges.py desde docs/claims_registry.yaml — NO editar a mano: "
                  "tests/test_dictionary_bridges.py falla si este fichero difiere de la regeneración. -->")

# (puente, de dónde a dónde, filas del registro que lo sostienen, nota breve)
BRIDGES = [
    ("δ₀", "imperfección primordial → todo el tramo pre-geométrico y δ_H del empalme", ["potencial-basal", "empalme-wkb", "circulo-delta0"],
     "δ_H = 0.0554 medido por el empalme C¹ (paisaje completo); el atractor δ₀* de Victoria es condicional a W_max"),
    ("f (fracción descargada) y T₀", "Potencial Basal → reloj S (S ≡ f)", ["potencial-basal", "s-clock-simulador-consistencia"],
     "S = f − f₀ + W/T₀ comprobado a 2e-13 sobre la trayectoria; T₀ = c̄δ₀³[1 + κ₁√δ₀ + …]"),
    ("Φ_ten (Florencia)", "estado entregado en S = 1,001 → N = e^{Φ_ten} de la cosmología", ["s-clock-simulador-consistencia", "diccionario-primordial-cosmologico"],
     "el reloj entrega Φ_ten = 0 DECLARADO (DECLARED_FORMS); la cosmología no lo lee (readable_by_cosmology = False)"),
    ("ε_K = λ_K − 1 (Residuos)", "Gea/Atlas → G_cosmo/G_N (BBN)", ["gea-newton-atlas", "residuos", "bbn-g-empress"],
     "ε_K = 0.012 viene de la Conj. 9.6 (−1.8 %), no de δ₀; la conexión ε_K = f(δ₀, λ_K, ξ, …) no existe (propuesta v36 §VIII)"),
    ("ξ", "acción de Gea → c_T² = ξ", ["mu-eta-atlas", "atlas-gauge-invariant"],
     "ξ = 1 fijado por GW170817: constante externa, no derivada del tramo pre-geométrico"),
    ("λ_K, α_a", "acción de Gea → µ_Atlas, c_s², PPN", ["mu-eta-atlas", "atlas-ppn", "atlas-gauge-invariant"],
     "ventana λ_K > 1, α_a ≤ 8e-7 (PPN): cotas, no valores derivados"),
    ("A (amplitud de Cronos) y ρ*", "Ley de Cronos débil → dinámica galáctica", ["cronos-amplitud-unica", "jeans-dsph", "oort-kz-cota-cronos", "diccionario-epsilon-c-saturante"],
     "A_Sculptor calibrada por el problema inverso 5A/5B; excluida en el plano (letra C, bytes no descargados); ρ* acotado al 58 % del plano"),
    ("S_post (unidad de S tras Florencia)", "reloj S → cronología post-geométrica", ["diccionario-unidad-S-post-florencia"],
     "ecuación 2 declarada: dS_post = Σ̇_post·dσ/T_sellada, con Σ̇_post y T_sellada nombradas y no derivadas"),
    ("C(S) = d ln a/dS", "corriente de conversión κ → ley de expansión (A.7)", ["diccionario-unidad-S-post-florencia", "vacio-2d-conversion-y-rotacion", "mapa-s-z-convencion"],
     "ecuación 3 declarada; κ̂ es calibración del brazo (ii) del vacío 2D (INDETERMINADO); S_hoy = 95 sigue siendo convención"),
    ("T(S) = dt_rel/dS", "Ley de Cronos de fondo → tiempo cosmológico (A.7)", ["mapa-s-z-convencion"],
     "t_rel = C·(S − S_birth)^α con α = 1: convención operativa LEGACY_V32"),
    ("S_hoy", "mapa S(z) → escalones f_id(S) de los canales", ["mapa-s-z-convencion"],
     "S_today = 95 (LEGACY_V32): convención declarada, sin derivación"),
    ("Ω_id,0, ε_Λ, z_trans", "canales ECV/MCV → fondo (A.4–A.6)", ["canales-oscuros", "fondo-cosmologico", "ajustes-produccion"],
     "parámetros del ajuste (F.3); ε_Λ no identificable con DESI DR2 ni Dovekie real"),
    ("α_n, S_n^post (escalones)", "colapsos post-Florencia → f_id(S) (A.5)", ["canales-oscuros"],
     "el tratado no fija valores: parámetros del ajuste; α = () apaga los escalones"),
    ("κ_lat, η_lat", "canal latente → w_lat(z) (A.6)", ["canales-oscuros"],
     "calibrados en el ajuste de producción"),
]


def load_registry() -> dict:
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))


def classify(rows: list[dict]) -> str:
    statuses = {r["status"] for r in rows}
    evidence = " ".join(str(r.get("evidence_level", "")) for r in rows).lower()
    dataset = " ".join(str(r.get("dataset", "")) for r in rows).lower()
    if statuses & {"hueco-declarado", "claim-no-derivado", "experimento-no-ejecutado"}:
        return "ausente"
    if "convención" in evidence or "convencion" in evidence:
        return "convencional"
    if "calibrado" in statuses or "calibrad" in evidence or "ancla externa" in dataset:
        return "calibrado"
    if "condicional" in statuses:
        return "condicional"
    return "derivado"


def render(reg: dict) -> str:
    by_id = {r["claim_id"]: r for r in reg["claims"]}
    lines = ["# Diccionario primordial → cosmológico: la tabla de puentes", "", GENERATED_NOTE, "",
             "Cada puente es una magnitud que el tramo pre-geométrico debería entregar a la cosmología (o una constante que la "
             "cosmología usa y debería salir de él). Su estatuto se LEE de las filas de `docs/claims_registry.yaml` que lo "
             "sostienen, con la regla de `scripts/make_dictionary_bridges.py` (ausente ≻ convencional ≻ calibrado ≻ condicional ≻ "
             "derivado). Es una vista, no un juicio: cambia cuando cambian las filas. Ninguna celda «derivado» es demostración "
             "física (E8); las cuatro ecuaciones del diccionario (1: forma de ε_c; 2: unidad de S post-Florencia; 3: C(S) desde κ; "
             "4: qué mide S) quedan donde el registro las tiene.", "",
             "| Puente | De → a | Estatuto | Filas | Nota |", "|---|---|---|---|---|"]
    counts: dict[str, int] = {}
    for name, span, ids, note in BRIDGES:
        rows = []
        for cid in ids:
            if cid not in by_id:
                raise SystemExit(f"FALLO CERRADO: la fila {cid} citada por el puente «{name}» no existe en el registro")
            rows.append(by_id[cid])
        kind = classify(rows)
        counts[kind] = counts.get(kind, 0) + 1
        lines.append(f"| {name} | {span} | **{kind}** | " + ", ".join(f"`{c}`" for c in ids) + f" | {note} |")
    lines += ["", "## Recuento", ""] + [f"- **{k}**: {counts.get(k, 0)}" for k in ("derivado", "condicional", "calibrado", "convencional", "ausente")]
    lines += ["", "## Conexión declarada de la unidad de S (ecuaciones 2 y 3)", "",
              "`cosmology.dark_channels.S_of_z_declared` devuelve el mapa S(z) vigente junto con la procedencia de su unidad: la "
              "convención S_hoy = 95 (LEGACY_V32) y el estatuto de `core.s_post_unit` (E2 y E3 declaradas, Σ̇_post y T_sellada no "
              "derivadas, κ pendiente). La cosmología CONSUME así la unidad de S de Florencia en modo declarado: el número no cambia; "
              "cambia lo que el artefacto dice de él."]
    return "\n".join(lines) + "\n"


def main() -> int:
    text = render(load_registry())
    if "--check" in sys.argv:
        return 0 if OUT.read_text(encoding="utf-8") == text else 1
    OUT.write_text(text, encoding="utf-8")
    print(f"{len(BRIDGES)} puentes → {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
