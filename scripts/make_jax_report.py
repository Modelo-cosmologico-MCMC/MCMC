#!/usr/bin/env python
"""Informe del crosscheck JAX — SOLO desde crosscheck.json;
idempotente, sin recomputar nada.

Uso: python scripts/make_jax_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = (Path(__file__).resolve().parent.parent / "results"
       / "2026-08-19_jax_crosscheck")


def main() -> None:
    d = json.loads((OUT / "crosscheck.json").read_text("utf-8"))
    s, g = d["summary"], d["gates"]
    b5 = d["block5_sne_production"]
    chi_lines = "\n".join(
        f"| {k} | {v['chi2_numpy']:.9f} | {v['chi2_jax']:.9f} | "
        f"{v['abs_diff']:.2e} |"
        for k, v in d["block4_chi2"].items())

    (OUT / "report.md").write_text(f"""# Crosscheck JAX ↔ NumPy del fondo (6A, prioridad 3)

Commit: `{d["code_commit"][:12]}`. Implementación independiente
(`validation/jax_background.py`: fórmulas del Apéndice A + mapa S↔z
operativo, integración Gauss-Legendre espectral de 96 nodos, float64;
únicas piezas compartidas: las fórmulas declaradas y las constantes
físicas de `SHARED_CONSTANTS`) contra las implementaciones NumPy de
producción. Estatuto: validación interna de implementación (E8) —
equivalencia numérica, no validación física.

## Puertas predeclaradas y resultado

max |ΔX/X| < {d["gates_declared"]["max_rel"]:.0e} y RMS <
{d["gates_declared"]["rms_rel"]:.0e} por bloque; |Δχ²| <
{d["gates_declared"]["chi2_abs"]:.0e}. Bloque 5 SIN puerta (medición
publicada).

| bloque | max | RMS | puerta |
|---|---|---|---|
| 1 — E, H, D_H (malla z ∈ [0, 3], 4 θ) | {s["block1_max"]:.2e} | {s["block1_rms"]:.2e} | {"PASS" if g["block1_background"] else "FAIL"} |
| 2 — mapa S↔z (S ∈ [1.001, 95]) | {s["block2_max"]:.2e} | {s["block2_rms"]:.2e} | {"PASS" if g["block2_s_map"] else "FAIL"} |
| 3 — vector DESI (13 comp., 4 θ) | {s["block3_max"]:.2e} | {s["block3_rms"]:.2e} | {"PASS" if g["block3_desi_vector"] else "FAIL"} |
| 4 — χ² (argmin publicados + rincón) | {s["block4_max_chi2_diff"]:.2e} | — | {"PASS" if g["block4_chi2"] else "FAIL"} |

**{"TODAS LAS PUERTAS PASS" if d["all_gates_pass"] else "PUERTAS SIN PASS — publicado tal cual"}.**

Los θ del bloque 3-4 incluyen los argmin PUBLICADOS de
benchmark.json y contrast.json — el crosscheck ata las dos
implementaciones exactamente donde viven los números publicados:

| θ | χ² NumPy | χ² JAX | \\|Δχ²\\| |
|---|---|---|---|
{chi_lines}

## El hallazgo del crosscheck (bloque histórico)

La primera ejecución midió el integrador del vector DESI de
producción (trapecio + interpolación lineal) en ~4.9e-8 relativo —
POR ENCIMA de la puerta — con efecto ≤ 2.7e-6 en los χ² publicados
(veredicto 6A robusto: ningún número con 3 decimales se movía). En
vez de relajar la puerta, se corrigió el integrador (cumulativa
Simpson O(h⁴) + cola Gauss-Legendre por bin; verificado a 5.8e-15
contra scipy quad) y se regeneraron los artefactos DESI. El número de
esta tabla es el del integrador corregido.

## Bloque 5 — distancias SNe de producción (sin puerta)

El integrador SNe de producción (`comoving_distance`, trapecio de
2048 puntos, usado por los ajustes v1/v2 ya publicados) difiere del
espectral en **{b5["comoving_distance"]["max_rel"]:.2e} relativo**
(máx sobre z ∈ [0.01, 2.26] en el θ de producción), es decir
≤ {b5["distance_modulus_max_abs_mag"]:.1e} mag en μ — despreciable
frente a σ_μ ~ 0.1 de Pantheon+ y sin efecto en ΔAIC/ΔBIC publicados.
Se publica como medición, no se toca la maquinaria de una ronda ya
fusionada; si una ronda futura necesita μ a < 1e-8, este número dice
exactamente qué cambiar.
""", encoding="utf-8")
    print(f"Informe: {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
