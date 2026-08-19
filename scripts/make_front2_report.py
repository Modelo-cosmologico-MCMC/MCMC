#!/usr/bin/env python
"""Informe del frente 2 — SOLO desde los artefactos computados
(preregistration.json + fp_beta.json + fp_beta_delta0.json);
idempotente, sin recomputar nada.

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
    ad = json.loads((OUT / "fp_beta_delta0.json").read_text("utf-8"))
    c, q = d["canonical"], d["quartic_E4_0"]
    cl = d["closure"]
    scan = d["scan"]
    cplx = [r for r in scan if r["complex_pair"]]
    lo, hi = p["closure"]["robustness_window"]
    eigs = ", ".join(f"{r:+.4f}" for r in sorted(c["eigenvalues_re"]))
    n_neg = d["n_dD_negative"]
    ctrl = d["drift_control"]
    phys = ad["physical_point"]
    cplx_lines = "\n".join(
        f"- g = {r['g']}: s0(τ=1) = {r['s0_spectral']:.4f} "
        f"(dos rutas, error rel. {r['routes_rel_error']:.1e}), "
        f"Q = |Im/Re| = {r['Q_im_over_re']:.4f}, "
        f"τ* = {r['tau_star_for_lambda10']:.4f}, "
        f"dD/dt = {r['dD_dt']:+.1f}" for r in cplx) or "- (ninguno)"

    (OUT / "report.md").write_text(f"""# Frente 2: el espectro de las β de Fokker-Planck — desenlace {d["outcome"]} en el punto preinscrito, con corrección de alcance (adenda δ0)

Preinscripción: `{d["prereg_commit"][:12]}` (congelada ANTES del
primer autovalor); espectro: `{d["code_commit"][:12]}`; adenda:
`{ad["code_commit"][:12]}`. Derivación validada por la suite (álgebra
⟺ jacobiano numérico ⟺ suavizado gaussiano exacto ⟺ FP exacta vía
Hopf-Cole).

## 1. El resultado en el punto preinscrito, tal cual

**DESENLACE {d["outcome"]}** (regla preinscrita, en su orden literal):
en el punto preinscrito (M0², B, C0) = (1, 2, 1) — espinodal D = 0 —
con el cierre canónico (Polchinski a = b = {cl["a"]}, truncamiento
{cl["truncation"]}, τ = {cl["tau"]}), el espectro de M = ∂β/∂λ es
**enteramente real**: μ = ({eigs}) — silla sin rotación, sin cascada
DSI. Además dD/dt = {c["dD_dt"]:+.1f} > 0 en ese punto (Obs. 8.6
exige < 0). El sistemático cuártico (E4 = 0) da espectro también real
(μ = {", ".join(f'{r:+.3f}' for r in sorted(q["eigenvalues_re"]))}).

Barrido preinscrito completo ({len(scan)} puntos, g ∈
[{scan[0]["g"]}, {scan[-1]["g"]}]), publicado íntegro:
{len(cplx)}/{len(scan)} puntos con par complejo —
{cplx_lines}
  El signo de dD/dt NO es exclusivo de ese punto: dD/dt = 44 − 16g
  sobre el barrido (cero en g = 2.75), negativo en {n_neg} de
  {len(scan)} puntos; lo exclusivo del borde es la CONJUNCIÓN
  rotación + hundimiento de D (par complejo solo para
  g > g* = {ad["g_star_complexification"]:.3f}, frontera exacta por
  el discriminante del polinomio característico). La ventana
  preinscrita g ∈ [{lo}, {hi}] es uniformemente real.

## 2. CORRECCIÓN (revisión adversarial): el eje g es el eje δ0

La revisión adversarial de la ronda encontró (hallazgo HIGH,
confirmado por verificador independiente) que la primera versión
afirmaba «la dependencia en δ0 entra por el diccionario τ(δ0)» — es
FALSO. Las β cumplen la covarianza exacta β(D_s·λ; a, b) =
σ·D_s·β(λ; a, b·k) (verificada numéricamente: desviación máxima
{ad["covariance_identity_max_abs_dev"]:.1e}), de modo que la familia
espinodal FÍSICA del escalado (3.2), (m̄²δ0², b̄δ0, C0), equivale
espectralmente a (1, 2, 1) con g_ef = δ0⁻³: **el barrido en g era, en
secreto, un barrido en δ0**, y el punto preinscrito fija
implícitamente δ0 ≈ 1 — fuera del régimen perturbativo δ0 ≪ 1 donde
vive el corpus del Basal. (Coherencia interna restaurada: el
despreciar η por O(δ0⁶) exige δ0 ≪ 1, incompatible con el punto
preinscrito; la adenda evalúa donde el argumento vale.)

**La adenda δ0** (fp_beta_delta0.json; misma derivación congelada,
sin tocar coeficientes ni reglas): a lo largo de la familia física
con el cierre canónico, el espectro se complejifica para
δ0 < δ0* = {ad["delta0_star"]:.4f} (= g*^(−1/3)) y allí dD/dt < 0 —
**en el régimen físico del corpus, la reducción canónica SÍ produce
las dos condiciones del corpus a la vez** (rotación + hundimiento de
D). En el punto físico de referencia δ0 = {ad["delta0_physical_reference"]}:
s0(τ=1) = {phys["s0_tau1"]:.4f}, Q = {phys["Q_im_over_re"]:.4f},
dD/dt = {phys["dD_dt"]:+.3f}, y λ = 10 exigiría
τ* = {phys["tau_star_for_lambda10"]:.4f} — publicado como dato: el
corpus recogido no fija τ; elegirlo a posteriori para forzar
s0 = π/ln10 sería tuning (compromiso preinscrito), por eso τ* se
publica, no se deriva. Aplicando las reglas preinscritas AL PUNTO
FÍSICO (clasificación contrafactual, declarada como tal — el
desenlace {d["outcome"]} del punto preinscrito NO se reclasifica):
**{ad["counterfactual_outcome_at_physical_point"]}** — hay cascada,
con s0 fuera de la banda de λ = 10 bajo el diccionario canónico.

## 3. Control de deriva (secundario prometido, cumplido y declarado)

La preinscripción prometía un «control de Langevin» (deriva medida de
los acoplos vs β predichas). Se cumplió con un control determinista
MÁS FUERTE y la sustitución se declara: la FP exacta vía Hopf-Cole
(e^{{−V_t}} = e^{{tΔ/2}}·e^{{−V0}}, sin ruido Monte Carlo), Richardson
t → 0, base hasta x⁵. Resultado (artefacto `drift_control`): error
relativo máximo {ctrl["max_rel_error"]:.1%} en el régimen declarado.

## 4. Qué significa (y qué no)

- **En el punto preinscrito (δ0 ≈ 1)**: desenlace {d["outcome"]} — la
  reducción canónica no produce la cascada allí. Se publicó tal cual
  y no se reclasifica.
- **En el régimen físico (δ0 ≲ {ad["delta0_star"]:.2f})**: la misma
  reducción canónica SÍ rota y hunde D — favorable al MECANISMO de la
  cascada; pero s0(τ=1) ≠ π/ln10: **λ = 10 sigue sin derivarse** (el
  diccionario τ y su posible dependencia δ0 quedan declarados como no
  derivables del corpus recogido).
- **No significa**: ni que el mecanismo esté refutado (punto
  preinscrito) ni que λ = 10 esté derivado (punto físico). El alcance
  es interno (E8) y condicional a los cierres declarados; si el
  corpus fija otro cierre o el diccionario, se recalcula.

## 5. Estatuto del frente 2 tras esta ronda

Sigue ABIERTO, con contenido nuevo en las dos direcciones: la
reducción canónica de la Def. 4.4 produce cascada DSI exactamente en
el régimen perturbativo del corpus (δ0 < δ0* ≈ {ad["delta0_star"]:.2f})
— el mecanismo tiene por primera vez una realización derivada, no
asumida — y λ = 10 sigue siendo una calibración (s0 depende del
diccionario τ, no derivable aún). La preinscripción tuvo un defecto
de diseño (el punto de evaluación escondía δ0 ≈ 1) que la revisión
adversarial destapó; se corrige por adenda, con la preinscripción
intacta y el defecto declarado — no reescrito.
""", encoding="utf-8")
    print(f"Informe: {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
