# Primera aplicación a Dovekie real — desenlace **INDETERMINADO**

Preinscripción `f630834326c0` (congelada antes de la primera lectura del HD); PASS de mocks `3ccdca260152`; ejecución commit `3dea1ef51`, 576.5 s.

**Lectura obligatoria del desenlace**: ningún desenlace preinscrito se cumple; se publica como indeterminado sin ajuste.

## Benchmark SN-only ΛCDM (like-for-like con el chain oficial)

Ω_m propio = 0.3305 (+0.0154 −0.0150) frente a oficial 0.3306 ± 0.0154: **Δ = -0.005σ_oficial**.

## Contraste conjunto Dovekie + CC + BAO

| brazo | χ²_min | χ²_ν | Ω_m | H0 | ε | AIC | BIC | conv |
|---|---|---|---|---|---|---|---|---|
| lcdm_statsys | 1656.29 | 0.893 | 0.3295 ± 0.0137 | 67.77 ± 0.61 | ≡ 0 | 1660.29 | 1671.34 | sí |
| mcmc_statsys | 1656.24 | 0.894 | 0.3288 ± 0.0139 | 67.76 ± 0.59 | +0.0184 ± 0.0371 | 1664.24 | 1686.35 | sí |
| lcdm_statonly | 1715.13 | 0.925 | 0.3464 ± 0.0107 | 67.28 ± 0.54 | ≡ 0 | 1719.13 | 1730.18 | sí |
| mcmc_statonly | 1714.98 | 0.926 | 0.3467 ± 0.0104 | 67.27 ± 0.53 | +0.0120 ± 0.0374 | 1722.98 | 1745.08 | sí |

**STAT+SYS (principal)**: ΔAIC(MCMC−ΛCDM) = +3.96, ΔBIC = +15.01, Δχ²_min = +0.042; ε = +0.0184 ± 0.0371, CI95 [-0.0442, +0.0905] (0 ∈ CI95: True); χ²_ν(ΛCDM) = 0.893.

**STATONLY (robustez)**: ΔAIC(MCMC−ΛCDM) = +3.85, ΔBIC = +14.90, Δχ²_min = +0.155; ε = +0.0120 ± 0.0374, CI95 [-0.0453, +0.0889] (0 ∈ CI95: True); χ²_ν(ΛCDM) = 0.925.

## Regla congelada aplicada (orden C → B → A → INDETERMINADO)

`{"C_bench_sigma": 3.0, "C_chi2nu_max": 1.3, "B_dBIC_pro_mcmc": 2.0, "A_dBIC_min": 0.0, "A_eps_sd_min": 0.04, "A_bench_sigma": 1.0}`

Roles: Dovekie real = primera aplicación · Unite = benchmark armonizado, NO replicación independiente · Union3 = contraste externo · Pantheon+ = disección. Esta corrida no confirma nada que Unite ya contenga. Estatuto: comprobación interna del pipeline sobre datos reales bajo preinscripción (E8) — no demostración física.
