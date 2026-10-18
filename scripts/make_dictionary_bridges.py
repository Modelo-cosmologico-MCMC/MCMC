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

# Huecos compartidos: dos puentes ausentes que son caras del MISMO hueco cuentan como un solo hueco en el recuento
# (revisión del autor del 30-sep). El puente sigue en la tabla con su estatuto; el grupo solo cambia el recuento de huecos.
GAP_GROUPS = {
    "ecuación 4": ("qué mide S tras Florencia. S_post y C(S) son el mismo hueco: Σ̇_post, T_sellada y κ son tres caras de la "
                   "corriente de descarga post-Florencia, y cerrar una cierra las tres (revisión del autor, 30-sep). Orden de "
                   "trabajo propuesto por el autor para cuando se ordene: la unidad de S post-Florencia desde el depósito sellado "
                   "(ecuación 2) y de ella C(S) (ecuación 3). La ecuación 4 no se ejecuta sin orden."),
}

# (puente, de dónde a dónde, filas del registro que lo sostienen, nota breve, grupo de hueco o None)
BRIDGES = [
    ("δ₀", "imperfección primordial → todo el tramo pre-geométrico y δ_H del empalme", ["potencial-basal", "empalme-wkb", "circulo-delta0"],
     "δ_H = 0.0554 medido por el empalme C¹ (paisaje completo); el atractor δ₀* de Victoria es condicional a W_max", None),
    ("f (fracción descargada) y T₀", "Potencial Basal → reloj S (S ≡ f)", ["potencial-basal", "s-clock-simulador-consistencia"],
     "S = f − f₀ + W/T₀ comprobado a 2e-13 sobre la trayectoria; T₀ = c̄δ₀³[1 + κ₁√δ₀ + …]", None),
    ("Φ_ten (Florencia, fondo)", "estado entregado en S = 1,001 → lapso N = e^{Φ_ten} del fondo", ["phi-ten-fondo-ley-cronos", "s-clock-simulador-consistencia"],
     "Def. 6.4: N = r/r̄ ≡ 1 en el fondo homogéneo ⟹ Φ_ten = 0 exacto (derivación del autor, 30-sep); el reloj lo calcula y lo entrega", None),
    ("Φ_ten (perturbaciones)", "r ≠ r̄ → modo escalar de Atlas (Def. 9.4)", ["gea-newton-atlas"],
     "Φ_ten ≠ 0 solo con r ≠ r̄: el modo escalar de Atlas, sano para λ_K ↓ 1⁺ y clasificado condicional (§13.4)", None),
    ("ε_K = λ_K − 1 (Residuos)", "Gea/Atlas → G_cosmo/G_N (BBN)", ["gea-newton-atlas", "residuos", "bbn-g-empress"],
     "ε_K = 0.012 viene de la Conj. 9.6 (−1.8 %), no de δ₀; la conexión ε_K = f(δ₀, λ_K, ξ, …) no existe (propuesta v36 §VIII)", None),
    ("ξ", "acción de Gea → c_T² = ξ", ["mu-eta-atlas", "atlas-gauge-invariant"],
     "ξ = 1 fijado por GW170817: constante externa, no derivada del tramo pre-geométrico", None),
    ("λ_K, α_a", "acción de Gea → µ_Atlas, c_s², PPN", ["mu-eta-atlas", "atlas-ppn", "atlas-gauge-invariant"],
     "ventana λ_K > 1, α_a ≤ 8e-7 (PPN): cotas, no valores derivados", None),
    ("A (amplitud de Cronos) y ρ*", "Ley de Cronos débil → dinámica galáctica", ["cronos-amplitud-unica", "jeans-dsph", "oort-kz-cota-cronos", "diccionario-epsilon-c-saturante"],
     "A_Sculptor calibrada por el problema inverso 5A/5B; excluida en el plano (letra C, bytes no descargados); ρ* acotado al 58 % del plano", None),
    ("S_post (unidad de S tras Florencia)", "reloj S → cronología post-geométrica", ["diccionario-unidad-S-post-florencia"],
     "ecuación 2 declarada: dS_post = Σ̇_post·dσ/T_sellada, con Σ̇_post y T_sellada nombradas y no derivadas", "ecuación 4"),
    ("C(S) = d ln a/dS", "corriente de conversión κ → ley de expansión (A.7)",
     ["diccionario-unidad-S-post-florencia", "vacio-2d-conversion-y-rotacion", "beta-extra-cribado", "mapa-s-z-convencion"],
     "ecuación 3 declarada; κ no ganó estatuto: la corriente no cruza la diagonal en S ≃ 1 (vacío 2D, INDETERMINADO) ni produce "
     "la Década (cribado de β, resultado negativo); S_hoy = 95 sigue siendo convención", "ecuación 4"),
    ("T(S) = dt_rel/dS", "Ley de Cronos de fondo → tiempo cosmológico (A.7)", ["mapa-s-z-convencion"],
     "t_rel = C·(S − S_birth)^α con α = 1: convención operativa LEGACY_V32", None),
    ("S_hoy", "mapa S(z) → escalones f_id(S) de los canales", ["mapa-s-z-convencion"],
     "S_today = 95 (LEGACY_V32): convención declarada, sin derivación", None),
    ("Ω_id,0, ε_Λ, z_trans", "canales ECV/MCV → fondo (A.4–A.6)", ["canales-oscuros", "fondo-cosmologico", "ajustes-produccion"],
     "parámetros del ajuste (F.3); ε_Λ no identificable con DESI DR2 ni Dovekie real", None),
    ("α_n, S_n^post (escalones)", "colapsos post-Florencia → f_id(S) (A.5)", ["canales-oscuros"],
     "el tratado no fija valores: parámetros del ajuste; α = () apaga los escalones", None),
    ("κ_lat, η_lat", "canal latente → w_lat(z) (A.6)", ["canales-oscuros"],
     "calibrados en el ajuste de producción", None),
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


KINDS = ("derivado", "condicional", "calibrado", "convencional", "ausente")


def table(reg: dict) -> list[dict]:
    """Los puentes con su estatuto leído del registro (la regla de `classify`); fallo cerrado si una fila no existe o si
    un puente con grupo de hueco no es ausente."""
    by_id = {r["claim_id"]: r for r in reg["claims"]}
    out = []
    for name, span, ids, note, gap in BRIDGES:
        rows = []
        for cid in ids:
            if cid not in by_id:
                raise SystemExit(f"FALLO CERRADO: la fila {cid} citada por el puente «{name}» no existe en el registro")
            rows.append(by_id[cid])
        kind = classify(rows)
        if gap is not None and (gap not in GAP_GROUPS or kind != "ausente"):
            raise SystemExit(f"FALLO CERRADO: el puente «{name}» declara el hueco «{gap}» pero su estatuto es {kind}")
        out.append({"bridge": name, "span": span, "rows": ids, "note": note, "gap": gap, "kind": kind})
    return out


def gap_count(tab: list[dict]) -> int:
    """Huecos distintos: los puentes ausentes sin grupo cuentan uno cada uno; los de un mismo grupo, uno en total."""
    absent = [b for b in tab if b["kind"] == "ausente"]
    return len({b["gap"] or f"__solo__{b['bridge']}" for b in absent})


def render(reg: dict) -> str:
    tab = table(reg)
    lines = ["# Diccionario primordial → cosmológico: la tabla de puentes", "", GENERATED_NOTE, "",
             "Cada puente es una magnitud que el tramo pre-geométrico debería entregar a la cosmología (o una constante que la "
             "cosmología usa y debería salir de él). Su estatuto se LEE de las filas de `docs/claims_registry.yaml` que lo "
             "sostienen, con la regla de `scripts/make_dictionary_bridges.py` (ausente ≻ convencional ≻ calibrado ≻ condicional ≻ "
             "derivado). Es una vista, no un juicio: cambia cuando cambian las filas. Ninguna celda «derivado» es demostración "
             "física (E8); las cuatro ecuaciones del diccionario (1: forma de ε_c; 2: unidad de S post-Florencia; 3: C(S) desde κ; "
             "4: qué mide S) quedan donde el registro las tiene.", "",
             "| Puente | De → a | Estatuto | Filas | Nota |", "|---|---|---|---|---|"]
    counts: dict[str, int] = {}
    for b in tab:
        counts[b["kind"]] = counts.get(b["kind"], 0) + 1
        gap = f" Hueco compartido: {b['gap']}." if b["gap"] else ""
        lines.append(f"| {b['bridge']} | {b['span']} | **{b['kind']}** | " + ", ".join(f"`{c}`" for c in b["rows"]) + f" | {b['note']}.{gap} |")
    lines += ["", "## Recuento", ""] + [f"- **{k}**: {counts.get(k, 0)}" for k in KINDS]
    groups = sorted({b["gap"] for b in tab if b["gap"]})
    lines += ["", f"Huecos distintos entre los puentes ausentes: **{gap_count(tab)}**."] + \
             [f"- **{g}**: {GAP_GROUPS[g]}" for g in groups]
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
