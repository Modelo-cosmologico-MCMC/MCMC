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


def ceil_2sig(x: float) -> str:
    """Cota superior a 2 cifras: redondeo hacia ARRIBA, para que todo
    «≤» publicado sea verdadero (hallazgo: dos cotas del informe
    estaban redondeadas por debajo del máximo medido)."""
    import math
    if x == 0.0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    m = math.ceil(x / 10 ** (e - 1)) / 10.0
    if m >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{m:.1f}e{e:+03d}"


def main() -> None:
    d = json.loads((OUT / "crosscheck.json").read_text("utf-8"))
    pre = json.loads((OUT / "crosscheck_pre_fix.json").read_text(
        "utf-8"))
    s, g = d["summary"], d["gates"]
    b5 = d["block5_sne_production"]
    iq = d["integrator_vs_quad"]
    pre_pub_max = max(
        v["abs_diff"] for k, v in pre["block4_chi2"].items()
        if k != "corner")
    pub_max = max(v["abs_diff"] for k, v in d["block4_chi2"].items()
                  if k != "corner")
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
benchmark.json y contrast.json; en ESOS argmin,
|Δχ²| ≤ {ceil_2sig(pub_max)} (el máximo del bloque,
{s["block4_max_chi2_diff"]:.2e}, corresponde al rincón de estrés
(ε = −0.05, z_trans = 1), que no es un argmin publicado):

| θ | χ² NumPy | χ² JAX | \\|Δχ²\\| |
|---|---|---|---|
{chi_lines}

## El hallazgo del crosscheck (bloque histórico)

La primera ejecución (registro `crosscheck_pre_fix.json`) midió el
integrador del vector DESI de producción (trapecio + interpolación
lineal) en ~{pre["summary"]["block3_max"]:.1e} relativo — POR ENCIMA
de la puerta — con efecto ≤ {ceil_2sig(pre_pub_max)} en los χ² de los
argmin publicados (veredicto 6A robusto: ningún número con 3
decimales se movía). En vez de relajar la puerta, se corrigió el
integrador (cumulativa Simpson O(h⁴) + cola Gauss-Legendre por bin) y
se regeneraron los artefactos DESI. La verificación del integrador
corregido contra scipy.quad está REGISTRADA en el artefacto
(`integrator_vs_quad`): máx {iq["max_rel_dm"]:.1e} relativo en DM
({iq["thetas"]}). Nota de procedencia (hallazgo declarado de la
revisión): el registro pre-fix se generó con el runner aún sin
commitear sobre el árbol de `{pre["code_commit"][:12]}` — su
code_commit NO contiene al generador (que llega en `0385851`); la
garantía de árbol limpio aplica a `crosscheck.json` y a los
artefactos DESI, y la reproducción del pre-fix combina cosmology/ de
`{pre["code_commit"][:12]}` con el runner de `0385851`.

## Bloque 5 — distancias SNe de producción (sin puerta)

El integrador SNe de producción (`comoving_distance`, trapecio de
2048 puntos, usado por los ajustes v1/v2 ya publicados) difiere del
espectral en **{b5["comoving_distance"]["max_rel"]:.2e} relativo**
(máx sobre z ∈ [0.01, 2.26] en el θ de producción), es decir
≤ {ceil_2sig(b5["distance_modulus_max_abs_mag"])} mag en μ — despreciable
frente a σ_μ ~ 0.1 de Pantheon+ y sin efecto en ΔAIC/ΔBIC publicados.
Se publica como medición, no se toca la maquinaria de una ronda ya
fusionada; si una ronda futura necesita μ a < 1e-8, este número dice
exactamente qué cambiar.
""", encoding="utf-8")
    print(f"Informe: {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
