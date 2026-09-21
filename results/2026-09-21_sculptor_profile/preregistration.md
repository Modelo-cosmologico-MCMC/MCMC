# Preinscripción — Perfil σ_los(R) de Sculptor: la forma que predice la amplitud única del 5E frente al perfil binado de Walker et al. 2009

Congelada 2026-09-21T14:11:20.513824+00:00 en el commit `5ea39a617` (que contiene el generador). sha256 de `preregistration.json`: `604f9c254c7e8283949137d23bb75fb4200b5341a369b1b89f06622ce53af002`.

## Declarado

- **frozen_setup**: montaje del 5E (dynamics.sculptor_transfer.frozen_config): Plummer con R_half = 260 pc, Υ⋆ = 2, A_Sculptor
- **A_sculptor**: 6.201588782789579e-07
- **beta_arm**: [-0.5, 0.0, 0.3]
- **dataset**: walker2009 (CDS J/AJ/137/3100, table4.dat: Sculptor, 1818 estrellas)
- **dataset_state_at_freeze**: DATA_UNAVAILABLE (CDS/VizieR 403 vía proxy, 21-sep-2026 14:10 UTC)
- **membership**: probabilidad de pertenencia ≥ 0.9 (columna del catálogo)
- **binning**: bins de igual número (≈ 150 estrellas) en R proyectado ≤ 1 kpc; σ_los por bin = desviación típica corregida por errores de medida (⟨e²⟩ restado); error del bin σ/√(2N)
- **shape_prediction**: pico central ≈ 2× lo observado y exterior newtoniano ÷3–4 (β = 0): la forma es la predicción congelada de jeans-dsph
- **observed_reference_not_ingested**: ≈ 9–10 km/s, plano hasta ≳ 1 kpc (Walker et al. 2007, 2009; Battaglia et al. 2008) — referencia, no ingerida

## Predicción congelada (σ_los en km/s; newtoniano / newtoniano + Cronos)

| R [pc] | β = −0.5 | β = 0 | β = +0.3 |
|---|---|---|---|
| 10 | 2.56 / 14.00 | 3.56 / 19.14 | 4.83 / 25.11 |
| 30 | 2.73 / 15.25 | 3.55 / 18.74 | 4.37 / 21.78 |
| 60 | 2.97 / 16.05 | 3.51 / 17.47 | 3.96 / 18.38 |
| 100 | 3.18 / 15.05 | 3.44 / 14.96 | 3.62 / 14.67 |
| 150 | 3.26 / 12.22 | 3.31 / 11.50 | 3.31 / 10.79 |
| 200 | 3.23 / 9.28 | 3.17 / 8.51 | 3.08 / 7.82 |
| 260 | 3.13 / 6.58 | 2.99 / 5.95 | 2.85 / 5.42 |
| 350 | 2.94 / 4.27 | 2.75 / 3.87 | 2.57 / 3.53 |
| 500 | 2.62 / 2.88 | 2.42 / 2.63 | 2.24 / 2.42 |
| 700 | 2.30 / 2.33 | 2.10 / 2.13 | 1.93 / 1.96 |
| 1000 | 1.96 / 1.96 | 1.79 / 1.79 | 1.64 / 1.64 |

## Reglas (congeladas)

- **chi2**: χ² = Σ_bins [(σ_pred(R_bin; β) − σ_obs,bin)/σ_err,bin]², ν = N_bins (ningún parámetro ajustado)
- **A**: χ²_ν ≤ 1.5 para algún β del brazo
- **B**: 1.5 < χ²_ν(mejor β) ≤ 3
- **C**: en los dos bins más internos (σ_pred − σ_obs)/σ_err > 3 para TODO β del brazo
- **INDETERMINADO**: walker2009 no AVAILABLE, N_bins < 4 o esquema no confirmado
- **verdict**: una letra; ninguna es afirmación sobre el tratado: sitúa la Ley de Cronos débil con A_Sculptor frente a la forma del perfil (E8). La normalización global reproduce σ_obs por construcción (A_Sculptor se obtuvo de ella): solo la FORMA es predicción

## Lo que no puede decidir

- la forma de ε_c(ρ) (diccionario)
- A (el 5E la fijó; aquí es hipótesis)
- la anisotropía real (β es brazo)
