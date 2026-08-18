# ESTADO: repetición con el fondo corregido (10-ago-2026)

Repetición del ajuste de producción v1 tras la corrección de
normalización del fondo (rama fix/background-normalization-desi):
H(0) = H0 exacto para todo parámetro admisible (transición normalizada
hoy, Ω_Λ_rel(z) = Ω_Λ0·F(z)/F(0)) y clausura plana por llamada
(Ω_DE,0 = 1 − Ω_m − Ω_r). En esta corrida ε es SOLO amplitud de la
transición — ya no arrastra el reescalado de H(0) que en la corrida
legacy estaba disponible solo para el brazo MCMC.

Condiciones: idénticas a la corrida legacy
(`results/2026-07-31_production_fit/`, etiquetada
legacy_pre_normalization) — mismos datos verificados por SHA-256
(31 CC + 6 BAO + 1371 SNe con covarianza completa, n = 1408), misma
semilla 42, misma configuración 32 walkers × 4000 pasos, mismos
priors (apéndice F). Convergencia por autocorrelación: OK
(τ ≤ 53, aceptación 0.57/0.72).

Resultado (informe completo en `production_fit_report.md`):

| | legacy (jul-2026) | corregido (10-ago-2026) |
|---|---|---|
| ΔAIC (MCMC − ΛCDM) | +4.03 | **+4.00** |
| ΔBIC (MCMC − ΛCDM) | +14.53 | **+14.50** |
| ε | 0.014 −0.039/+0.043 | 0.017 −0.039/+0.042 |
| H0 (MCMC) | sesgado | 67.87 ± 0.68 |
| Ω_m (MCMC) | sesgado | 0.326 ± 0.016 |

El veredicto diferencial pro-ΛCDM se confirma con el fondo corregido;
ε sigue compatible con 0. El resultado se publica sea cual sea
(contrato de honestidad). Estos son los valores citables vigentes del
ajuste v1.
