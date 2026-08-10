# ESTADO: repetición v2 con el fondo corregido (10-ago-2026)

Repetición del ajuste de producción v2 («la reconciliación», cinco
bloques) tras la corrección de normalización del fondo (rama
fix/background-normalization-desi): H(0) = H0 exacto y clausura plana
por llamada — ver `results/2026-08-10_production_fit/ESTADO.md` para
el detalle de la corrección y su asimetría en las corridas legacy.

Condiciones: idénticas a la corrida legacy
(`results/2026-08-01_production_fit_v2/`, etiquetada
legacy_pre_normalization) — mismos datos (31 CC + 6 BAO + 1371 SNe +
CMB comprimido (3) + 11 fσ8, n = 1422), misma semilla 42, misma
configuración 24 walkers × 8000 pasos, mismas salvedades declaradas
(z*/r_s por Hu–Sugiyama, CMB diagonal, r_d fiducial, M_B
marginalizada). Convergencia por autocorrelación: OK (τ ≤ 109,
aceptación 0.48/0.59).

Resultado (informe completo en `production_fit2_report.md`):

| | legacy (ago-2026) | corregido (10-ago-2026) |
|---|---|---|
| ΔAIC (MCMC − ΛCDM) | +4.01 | **+4.08** |
| ΔBIC (MCMC − ΛCDM) | +14.53 | **+14.60** |
| ε | 0.015 −0.039/+0.043 | 0.018 −0.041/+0.043 |
| H0 (MCMC) | sesgado | 68.19 ± 0.40 |
| Ω_m (MCMC) | sesgado | 0.312 ± 0.006 |
| σ8 (MCMC) | sesgado | 0.799 ± 0.029 |

El veredicto diferencial pro-ΛCDM se confirma con el fondo corregido
también con los cinco bloques; la ventaja del corpus (ΔBIC = −6.1)
sigue sin reaparecer. ε compatible con 0. Estos son los valores
citables vigentes del ajuste v2.
