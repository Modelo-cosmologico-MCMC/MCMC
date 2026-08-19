#!/usr/bin/env python
"""Informe del frente 2 — SOLO desde los artefactos computados
(preregistration.json + fp_beta.json); idempotente, sin recomputar
nada.

Uso: python scripts/make_front2_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = (Path(__file__).resolve().parent.parent / "results"
       / "2026-08-19_front2_fp_beta")


def main() -> None:
    p = json.loads((OUT / "preregistration.json").read_text("utf-8"))
    d = json.loads((OUT / "fp_beta.json").read_text("utf-8"))
    c, q = d["canonical"], d["quartic_E4_0"]
    cl = d["closure"]
    scan = d["scan"]
    cplx = [r for r in scan if r["complex_pair"]]
    lo, hi = p["closure"]["robustness_window"]
    eigs = ", ".join(f"{r:+.4f}" for r in sorted(c["eigenvalues_re"]))
    cplx_lines = "\n".join(
        f"- g = {r['g']}: s0(τ=1) = {r['s0_spectral']:.4f} "
        f"(dos rutas, error rel. {r['routes_rel_error']:.1e}), "
        f"Q = |Im/Re| = {r['Q_im_over_re']:.4f}, "
        f"dD/dt = {r['dD_dt']:+.1f}" for r in cplx) or "- (ninguno)"

    (OUT / "report.md").write_text(f"""# Frente 2: el espectro de las β de Fokker-Planck — desenlace {d["outcome"]}

Preinscripción: `{d["prereg_commit"][:12]}` (congelada ANTES del
primer autovalor); espectro: `{d["code_commit"][:12]}`. Derivación
validada por la suite (álgebra ⟺ jacobiano numérico ⟺ suavizado
gaussiano exacto ⟺ FP exacta vía Hopf-Cole).

## El resultado, tal cual

**DESENLACE {d["outcome"]}** (regla preinscrita): en el punto canónico
(espinodal D = 0, cierre Polchinski a = b = {cl["a"]}, truncamiento
{cl["truncation"]}, τ = {cl["tau"]}) el espectro de M = ∂β/∂λ es
**enteramente real**: μ = ({eigs}) — una silla sin rotación. La
reducción de Fokker-Planck de la Def. 4.4, con el cierre canónico,
**no produce el par complejo de la cascada DSI**: el exponente de
Victoria no emerge de esta β.

Doble negativa coherente con el corpus: además del espectro real,
**dD/dt = {c["dD_dt"]:+.1f} > 0** en la espinodal — el flujo canónico
SUBE el discriminante, cuando la Obs. 8.6 exige hundirlo (D debe
cruzar a negativo para disparar cada colapso).

## Robustez (todo publicado, caiga donde caiga)

- Ventana preinscrita g ∈ [{lo}, {hi}]: tipo espectral UNIFORME
  (real) — el desenlace A es robusto, no D.
- Barrido completo ({len(scan)} puntos, g ∈
  [{scan[0]["g"]}, {scan[-1]["g"]}]): {len(cplx)}/{len(scan)} puntos
  con par complejo:
{cplx_lines}
  El único punto complejo es el extremo de difusión dominante — y es
  también el ÚNICO donde dD/dt < 0: las dos condiciones del corpus
  (rotación + hundimiento de D) solo aparecen JUNTAS en el borde
  g ≫ 1, lejos del cierre canónico. Incluso allí, s0 no es π/ln10
  = {d["s0_target"]:.4f} sin un diccionario τ ajustado a mano
  (prohibido por la preinscripción).
- Sistemático del truncamiento: la variante cuártica (E4 = 0) da
  espectro también real (μ = {", ".join(f'{r:+.3f}' for r in sorted(q["eigenvalues_re"]))})
  — el desenlace no es un artefacto del cierre cúbico.

## Qué significa (y qué no)

- **Significa**: dentro de esta reducción (Polchinski d = 0 con kernel
  isótropo, jerarquía truncada, sector η despreciado), la cascada DSI
  NO es genérica: λ = 10 sigue siendo una CALIBRACIÓN (§14.2: «si
  resulta λ ≠ 10, el diez era convención» — pero aquí ni siquiera hay
  λ que resultar en el cierre canónico). El resultado ACOTA dónde
  podrían vivir unas β que sí roten: cierres fuertemente anisótropos
  o dominados por difusión (g ≳ 10), o física fuera de esta reducción.
- **No significa**: que el mecanismo del tratado esté refutado — el
  alcance preinscrito es interno (E8): UNA reducción con cierres
  declarados de la Def. 4.4, no la única posible. Si el corpus fija
  otro cierre (G anisótropa, temperatura entrópica dependiente de S,
  diccionario τ concreto), se recalcula con él — los cierres están
  expuestos como parámetros, no resueltos en silencio.

## Estatuto del frente 2 tras esta ronda

Sigue ABIERTO, con contenido nuevo: antes «las β no existen» (hueco
declarado); ahora «las β de la reducción canónica existen y NO
producen Victoria» (negativa concreta, con candados). La carga de la
derivación de λ = 10 queda sobre cierres no canónicos o sobre física
adicional del corpus — cuantificado, no vago.
""", encoding="utf-8")
    print(f"Informe: {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
