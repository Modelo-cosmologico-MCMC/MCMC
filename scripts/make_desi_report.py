#!/usr/bin/env python
"""Informe del frente DESI DR2 — generado SOLO desde los artefactos
computados (benchmark.json + contrast.json); idempotente, sin refits y
sin ningún cómputo en vivo (la revisión adversarial 6A encontró que la
primera versión computaba aquí un número puntual y lo publicaba como
cota del rango — todo número estructural vive ahora en contrast.json,
con candado).

Uso: python scripts/make_desi_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = (Path(__file__).resolve().parent.parent / "results"
       / "2026-08-18_desi_dr2_background")


def main() -> None:
    b = json.loads((OUT / "benchmark.json").read_text(encoding="utf-8"))
    c = json.loads((OUT / "contrast.json").read_text(encoding="utf-8"))
    eq, lb = b["equivalence"], b["lcdm_benchmark"]
    ident = eq["data_identity"]
    rows = c["rows"]
    st, pe = c["structural"], c["prior_eps"]
    all_row = next(r for r in rows if r["config"] == "DESI_ALL")

    tbl = "\n".join(
        f"| {r['config']} | {r['n']} | {r['chi2_lcdm']:.3f} | "
        f"{r['chi2_mcmc']:.3f}{'†' if r['at_boundary_mcmc'] else ''} | "
        f"{r['delta_chi2']:+.3f} | "
        f"{r['delta_aic']:+.3f} | {r['delta_bic']:+.3f} | "
        f"{r['eps_median']:+.4f} −{r['eps_minus']:.4f}/"
        f"+{r['eps_plus']:.4f} | {r['shift_omega_m_vs_ALL']:+.4f} |"
        for r in rows)
    dbic = [r["delta_bic"] for r in rows]
    dchi = [r["delta_chi2"] for r in rows]
    n_boundary = sum(1 for r in rows if r["at_boundary_mcmc"])
    max_shift = max(max(abs(r["shift_omega_m_vs_ALL"]) for r in rows),
                    max(abs(r["shift_omega_m_lcdm_vs_ALL"])
                        for r in rows))
    min_ratio = min(r["eps_width_ratio_vs_prior"] for r in rows)

    (OUT / "report.md").write_text(f"""# DESI DR2 BAO: benchmark ΛCDM y contraste con el fondo MCMC corregido (6A)

Commit del contraste: `{c["code_commit"][:12]}`; datos:
`data/manifests/desi_dr2_bao.json` (ingesta `bb0c1c9`, cargada con
`require_available` — manifest + estado + esquema + sha256 en tiempo
de ejecución). Contrato de la ronda: **datos ingeridos ≠ likelihood
validada ≠ modelo validado** — cada transición con artefacto y test.
Esta versión sustituye a la primera tras la revisión adversarial de la
ronda: fondo normalizado realmente fusionado (H(0) = H0, clausura
plana por llamada — `tests/test_physical_invariants.py`), χ²_min
acotado al soporte del prior, semilla real del sampler, identidad de
datos computada y números estructurales sobre el rango completo.
Regenerada además con el integrador del vector a O(h⁴) tras el
crosscheck JAX (ver `results/2026-08-19_jax_crosscheck/`): números
publicados idénticos a 3 decimales.

## 1. Likelihood validada (6A.2)

χ² propio ≡ Cobaya 3.6.2 (`bao.desi_dr2.desi_bao_all`, referencia
externa fijada): **máx |Δχ²| = {eq["max_abs_diff"]:.2e}** sobre
{eq["n_vectors"]} vectores sintéticos (tolerancia predeclarada
{eq["tolerance"]:.0e}); |logpdf de Cobaya en m = d| =
{abs(eq["logp_at_data"]):.1e}. Identidad de datos COMPUTADA (sha256
por fichero, offline): {ident["n_identical"]}/{ident["n_files"]}
ficheros del manifest idénticos a los que Cobaya lee en su packages
path (`bao_data` versión `{ident["packages_version"]}`) —
{ident["status"]}. El packages path se recrea con
`scripts/setup_cobaya_packages.py` (única pieza con red;
`COBAYA_PACKAGES_PATH` lo relocaliza).

## 2. Benchmark ΛCDM (6A.4) — la puerta

| | propio | oficial (arXiv:2503.14738) |
|---|---|---|
| Ω_m | {lb["Omega_m"]["median"]:.4f} −{lb["Omega_m"]["minus"]:.4f}/+{lb["Omega_m"]["plus"]:.4f} | 0.2975 ± 0.0086 |
| χ²_min (n=13, k=2) | {lb["chi2_min"]:.3f} | — |
| H0·r_d [km/s] | {lb["H0rd_kms"]["median"]:.0f} | — |

Con el fondo normalizado (clausura plana por llamada), Ω_m ES la
fracción de materia del ΛCDM plano — la comparación con el oficial es
entre cantidades homogéneas. Desviación |ΔΩ_m| =
{abs(lb["Omega_m"]["median"] - lb["official"]["Omega_m"]):.4f} =
**{lb["dev_sigma"]:.2f}σ oficial** ⟹ **PUERTA {lb["gate"]}** (regla
predeclarada: < 1σ). χ²_min por multistart acotado al soporte del
prior, argmin publicado en el artefacto. Solo tras este PASS corre el
contraste.

## 3. Contraste MCMC vs ΛCDM (6A.5) — todas las configuraciones

Convenciones idénticas (emcee 32×4000 en ambos modelos, semilla
global 42 — cadenas bit-reproducibles —, priors comunes, H0·r_d COMÚN:
el MCMC no deriva r_d; likelihood validada; mismo minimizador
acotado). Se publican TODAS las configuraciones preinscritas — sin
selección posterior:

| config | n | χ²_Λ | χ²_M | Δχ² | ΔAIC | ΔBIC | ε_Λ | ΔΩ_m vs ALL |
|---|---|---|---|---|---|---|---|---|
{tbl}

(Δ = MCMC − ΛCDM; positivo favorece a ΛCDM. † argmin del χ²_M en la
frontera del soporte del prior (ε = −0.05 y/o z_trans = 1) — ocurre en
{n_boundary} de 8 configuraciones y se publica en `contrast.json`, no
se oculta: el mínimo es del CIERRE del soporte declarado.)

**Lectura.** (i) El veredicto es UNIFORME: Δχ² ∈
[{min(dchi):+.3f}, {max(dchi):+.3f}] — los dos parámetros extra del
MCMC no compran ajuste que compense su coste — y ΔBIC ∈
[{min(dbic):+.2f}, {max(dbic):+.2f}] pro-ΛCDM en las 8; el mismo
veredicto que los ajustes de producción v1/v2 repetidos con el fondo
corregido (ΔBIC = +14.50/+14.60 con n = 1408/1422), ahora sobre DESI
DR2. (ii) Ningún bin es decisivo: quitar cualquiera (incluido
lrg-z1 ≡ el antiguo LRG2) mueve la mediana de Ω_m menos de
{max_shift:.4f} en AMBOS modelos y no cambia el signo de nada.
(iii) ε_Λ queda DOMINADO POR EL PRIOR, y la comparación es contra la
referencia correcta: el prior efectivamente muestreado es
N(0.012, 0.05²) TRUNCADO a (−0.05, 0.10), con mediana
{pe["median"]:+.4f} −{pe["minus"]:.4f}/+{pe["plus"]:.4f} — el cociente
anchura posterior / anchura prior es ≥ {min_ratio:.2f} en las 8
configuraciones. La razón es ESTRUCTURAL y está computada en el
artefacto (`contrast.json → structural`, con z_trans = {st["z_trans"]}
del corpus): con la transición normalizada hoy, ε = {st["eps_probe"]}
altera E(z) en ≤ {st["dE_rel_max_range"]:.1e} relativo sobre TODO el
rango BAO z ≤ 2.33 (máximo en z = {st["z_at_max"]:.2f}; en z = 2.33:
{st["dE_rel_at_z233"]:.1e}); la distancia χ² entre los vectores DESI
con y sin ε a parámetros fijos es {st["chi2_distance_fixed_params"]:.2e},
y tras reabsorber (Ω_m, H0·r_d) el residuo cae a
{st["chi2_residual_after_reabsorbing"]:.1e}. BAO DR2 constriñe la
FORMA de la expansión, y en esta parametrización el fondo MCMC es casi
degenerado con ΛCDM en ese rango; lo que el dato castiga es la
parsimonia. (iv) ε = {all_row["eps_median"]:+.4f}
−{all_row["eps_minus"]:.4f}/+{all_row["eps_plus"]:.4f} en DESI_ALL —
compatible con 0 y con el 0.012 del corpus (sin necesidad, no
excluida).

## Estatuto

condicional y parcial: (a) BAO-only con r_d como calibración común —
sin sector pre-recombinación, el MCMC no reclama r_d; (b) la
degeneración estructural (iii) significa que este frente NO es el
test sensible de ε_Λ en esta parametrización — los sensibles son los
que sondean z ≳ z_trans o la física de r_d (fuera del alcance
actual); el resultado se publica igualmente; (c) benchmark contra el
valor puntual oficial publicado — el cruce con cadenas/best-fits
oficiales completos queda pendiente del host bloqueado
(data.desi.lbl.gov); (d) próximos pasos del frente: SNe (mocks
Dovekie primero) y combinaciones DESI + {{Dovekie, Pantheon+, Union3}}
con el mismo contrato.
""", encoding="utf-8")
    print(f"Informe: {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
