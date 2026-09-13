# Canal Atlas de µ/η derivado desde la acción — desenlace **A** (E1 PASS, E2 PASS)

Preinscripción: `62ef5c724f6c` (congelada ANTES de ejecutar); ejecución: commit `5f8eea23e`; re-derivación simbólica en 125.0 s. Sin datos: teoría pura.

## E1 — identidades reproducidas desde la acción

| identidad | reproducida |
|---|---|
| sello_newton_reproduced | ✓ |
| delta_is_j0_plus_3phi | ✓ |
| mu_matches_closed_form | ✓ |
| eta_matches_closed_form | ✓ |
| mu_sub_is_1_over_1_minus_alpha_over_2xi | ✓ |
| eta_sub_is_1 | ✓ |
| gr_limit_mu_eta_1 | ✓ |
| G_growth_equals_G_local | ✓ |
| G_growth_is_GB_over_xi_minus_alpha_half | ✓ |
| cs2_matches_closed_form | ✓ |
| cT2_is_xi | ✓ |

- µ_QS(e) = `-2*xi/(alpha_a - 9*epsilon_H**2*lambda_K + 3*epsilon_H**2 - 2*xi)`
- η_QS(e) = `-(9*epsilon_H**2*lambda_K**2 - 9*epsilon_H**2*lambda_K*xi - 6*epsilon_H**2*lambda_K + 3*epsilon_H**2*xi + epsilon_H**2 + 2*lambda_K*xi - 2*xi)/(xi*(9*epsilon_H**2*lambda_K - 3*epsilon_H**2 - 2*lambda_K + 2))`
- G_growth/G_B = `-2/(alpha_a - 2*xi)` = G_local/G_B = `-2/(alpha_a - 2*xi)`
- c_s² = `-xi*(alpha_a - 2*xi)*(lambda_K - 1)/(alpha_a*(3*lambda_K - 1))`; coef. cinético = `(3*lambda_K - 1)/(kappa*(lambda_K - 1))`
- c_T² = `xi`
- η_QS en GR estricto a e finito: `1/3` — la degeneración 0/0 en λ_K = 1 (precisión (i)): el límite GR toma e → 0 primero

## E2 — integración numérica completa (sin QS)

Punto λ_K = 1.05, ξ = 1.02, α_a = 0.3: µ_sub = 1.172414, p_QS = 1.134178.

| k/H0 | e_fin | µ_num/µ_QS − 1 | η_num − 1 | cola QS de η (no puerta) | p_num | p_QS | std(p) | cambios de signo | veredicto |
|---|---|---|---|---|---|---|---|---|---|
| 66.667 | 1.00e-02 | +2.79e-04 | +2.97e-04 | +8.33e-03 | 1.1335 | 1.1342 | 5.5e-04 | 0 | PASS |
| 200 | 3.33e-03 | -5.27e-04 | +4.61e-04 | +9.16e-04 | 1.1341 | 1.1342 | 1.1e-04 | 0 | PASS |
| 600 | 1.11e-03 | +5.94e-05 | -3.28e-05 | +1.02e-04 | 1.1342 | 1.1342 | 5.9e-06 | 0 | PASS |

## Lectura

**Desenlace A**: la re-derivación desde la acción reproduce las formas cerradas y el sistema completo converge al orden dominante QS. El offset α_a/(2ξ) se cancela contra la G local: el canal Atlas no deja firma sub-horizonte en (µ, η) al orden dominante y la cola k² del canal Cronos queda como la ÚNICA firma sub-horizonte del sector perturbativo. ATLAS_STATUS pasa a DERIVADO-NULO.

**Precisión (ii)**: la cola QS de η lleva el polo 1/(λ_K−1) — coeficiente 45.3 en el punto del arnés y 173 con ε_K = 0.012 — y el arnés NO la reproduce (esperado: la QS descarta términos del mismo orden). Los coeficientes completos de las colas son frontera declarada (sector de velocidades); el parámetro pequeño efectivo es e/√(λ_K−1).

**Residuos refinados**: G_cosmo/G_local = (2ξ − α_a)/(3λ_K − 1) ≈ 1 − (3/2)ε_K − α_a/2 — BBN acota la combinación. **Erratum candidata H.2.2** (decisión del autor): c_s² diverge, no se anula, cuando α_a → 0 a λ_K fijo.

**Lo que NO queda derivado**: coeficientes de las colas; régimen superhorizonte; cotas PPN sobre (ε_K, α_a); acoplamiento fuerte.
