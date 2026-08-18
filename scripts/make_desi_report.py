#!/usr/bin/env python
"""Informe del frente DESI DR2 — generado SOLO desde los artefactos
computados (benchmark.json + contrast.json); idempotente, sin refits.

Uso: python scripts/make_desi_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = (Path(__file__).resolve().parent.parent / "results"
       / "2026-08-18_desi_dr2_background")


def main() -> None:
    b = json.loads((OUT / "benchmark.json").read_text(encoding="utf-8"))
    c = json.loads((OUT / "contrast.json").read_text(encoding="utf-8"))
    eq, lb = b["equivalence"], b["lcdm_benchmark"]
    rows = c["rows"]
    all_row = next(r for r in rows if r["config"] == "DESI_ALL")

    # lectura estructural: tamaño del efecto de la transición del
    # corpus en el rango BAO (z ≤ 2.33) — computado, no afirmado
    from cosmology.desi_background_fit import E_of_z
    z_ly = 2.33
    dE = abs(float(E_of_z(z_ly, 0.2975, eps=0.05))
             / float(E_of_z(z_ly, 0.2975, eps=0.0)) - 1.0)

    tbl = "\n".join(
        f"| {r['config']} | {r['n']} | {r['chi2_lcdm']:.3f} | "
        f"{r['chi2_mcmc']:.3f} | {r['delta_chi2']:+.3f} | "
        f"{r['delta_aic']:+.3f} | {r['delta_bic']:+.3f} | "
        f"{r['eps_median']:+.4f} −{r['eps_minus']:.4f}/"
        f"+{r['eps_plus']:.4f} | {r['shift_omega_m_vs_ALL']:+.4f} |"
        for r in rows)
    dbic = [r["delta_bic"] for r in rows]
    dchi = [r["delta_chi2"] for r in rows]

    (OUT / "report.md").write_text(f"""# DESI DR2 BAO: benchmark ΛCDM y contraste con el fondo MCMC corregido (6A)

Commit del contraste: `{c["code_commit"][:12]}`; datos:
`data/manifests/desi_dr2_bao.json` (ingesta `bb0c1c9`, cargada con
`require_available` — manifest + estado + esquema + sha256 en tiempo
de ejecución). Contrato de la ronda: **datos ingeridos ≠ likelihood
validada ≠ modelo validado** — cada transición con artefacto y test.

## 1. Likelihood validada (6A.2)

χ² propio ≡ Cobaya 3.6.2 (`bao.desi_dr2.desi_bao_all`, referencia
externa fijada): **máx |Δχ²| = {eq["max_abs_diff"]:.2e}** sobre
{eq["n_vectors"]} vectores sintéticos (tolerancia predeclarada
{eq["tolerance"]:.0e}); logpdf de Cobaya en m = d: 0 exacto.
Identidad de datos: {eq["data_identity"]}.

## 2. Benchmark ΛCDM (6A.4) — la puerta

| | propio | oficial (arXiv:2503.14738) |
|---|---|---|
| Ω_m | {lb["Omega_m"]["median"]:.4f} −{lb["Omega_m"]["minus"]:.4f}/+{lb["Omega_m"]["plus"]:.4f} | 0.2975 ± 0.0086 |
| χ²_min (n=13, k=2) | {lb["chi2_min"]:.3f} | — |
| H0·r_d [km/s] | {lb["H0rd_kms"]["median"]:.0f} | — |

Desviación |ΔΩ_m| = {abs(lb["Omega_m"]["median"] - 0.2975):.4f} =
**{abs(lb["Omega_m"]["median"] - 0.2975) / 0.0086:.2f}σ oficial** ⟹
**PUERTA {lb["gate"]}** (regla predeclarada: < 1σ). Solo tras este
PASS corre el contraste.

## 3. Contraste MCMC vs ΛCDM (6A.5) — todas las configuraciones

Convenciones idénticas (sampler, semilla, priors, H0·r_d COMÚN — el
MCMC no deriva r_d; likelihood validada). Se publican TODAS las
configuraciones preinscritas — sin selección posterior:

| config | n | χ²_Λ | χ²_M | Δχ² | ΔAIC | ΔBIC | ε_Λ | ΔΩ_m vs ALL |
|---|---|---|---|---|---|---|---|---|
{tbl}

(Δ = MCMC − ΛCDM; positivo favorece a ΛCDM.)

**Lectura.** (i) El veredicto es UNIFORME: Δχ² ∈
[{min(dchi):+.3f}, {max(dchi):+.3f}] — los dos parámetros extra del
MCMC no compran ajuste en ninguna configuración — y ΔBIC ∈
[{min(dbic):+.2f}, {max(dbic):+.2f}] pro-ΛCDM en las 8; el mismo
veredicto que los ajustes de producción v1/v2 (ΔBIC = +14.5 con
n = 1408), ahora sobre DESI DR2. (ii) Ningún bin es decisivo:
quitar cualquiera (incluido lrg-z1 ≡ el antiguo LRG2) mueve Ω_m
menos de {max(abs(r["shift_omega_m_vs_ALL"]) for r in rows):.4f} y
no cambia el signo de nada. (iii) ε_Λ queda DOMINADO POR EL PRIOR
(posterior ≈ N(0.012, 0.05) en todas las configuraciones): DESI DR2
BAO no mide ε con esta parametrización — razón ESTRUCTURAL,
computada: con z_trans = 8.9 y Δz = 1.5, la transición normalizada
hoy altera E(z ≤ 2.33) en ≤ {dE:.1e} relativo incluso con ε = 0.05
(el rango BAO está entero en la meseta post-transición). BAO DR2
constriñe la FORMA de la expansión, y en ese rango el fondo MCMC del
corpus es casi degenerado con ΛCDM; lo que el dato castiga es la
parsimonia. (iv) ε = {all_row["eps_median"]:+.4f}
−{all_row["eps_minus"]:.4f}/+{all_row["eps_plus"]:.4f} en DESI_ALL —
compatible con 0 y con el 0.012 del corpus (sin necesidad, no
excluida).

## Estatuto

condicional y parcial: (a) BAO-only con r_d como calibración común —
sin sector pre-recombinación, el MCMC no reclama r_d; (b) la
degeneración estructural (iii) significa que este frente NO es el
test sensible de ε_Λ — los sensibles son los que sondean z ≳ z_trans
o la física de r_d (fuera del alcance actual); el resultado se
publica igualmente; (c) benchmark contra el valor puntual oficial
publicado — el cruce con cadenas/best-fits oficiales completos queda
pendiente del host bloqueado (data.desi.lbl.gov); (d) próximos pasos
del frente: SNe (mocks Dovekie primero) y combinaciones
DESI + {{Dovekie, Pantheon+, Union3}} con el mismo contrato.
""", encoding="utf-8")
    print(f"Informe: {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
