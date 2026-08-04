# Sensibilidad de la naturalidad de δ_H a los priors del paisaje

Barrido con semilla 20260804, n = 20000 paisajes por configuración: priors (uniforme, log-uniforme, normal truncada) × dominios O(1) × extensión de b̄ × filtro fértil. Tarea nacida de la auditoría v35.1: condicionar la afirmación de naturalidad de la Nota I (§3.2).

| prior | dominio | b̄×2 | fértil | mediana | [p5, p95] | mín | cota |
|---|---|---|---|---|---|---|---|
| uniform | (0.5, 2) | sí | sí | 0.0540 | [0.0368, 0.1146] | 0.0332 | 0.0330 |
| uniform | (0.333, 3) | sí | sí | 0.0365 | [0.0237, 0.0994] | 0.0218 | 0.0217 |
| uniform | (0.25, 4) | sí | sí | 0.0277 | [0.0176, 0.0842] | 0.0163 | 0.0163 |
| loguniform | (0.5, 2) | sí | sí | 0.0620 | [0.0368, 0.1493] | 0.0332 | 0.0330 |
| loguniform | (0.333, 3) | sí | sí | 0.0508 | [0.0241, 0.1850] | 0.0218 | 0.0217 |
| loguniform | (0.25, 4) | sí | sí | 0.0455 | [0.0182, 0.2206] | 0.0163 | 0.0163 |
| normal | (0.5, 2) | sí | sí | 0.0634 | [0.0409, 0.1104] | 0.0332 | 0.0330 |
| normal | (0.333, 3) | sí | sí | 0.0428 | [0.0264, 0.0849] | 0.0218 | 0.0217 |
| normal | (0.25, 4) | sí | sí | 0.0322 | [0.0196, 0.0679] | 0.0163 | 0.0163 |
| uniform | (0.5, 2) | sí | no | 0.0546 | [0.0370, 0.1102] | 0.0333 | 0.0330 |
| uniform | (0.5, 2) | no | sí | 0.1130 | [0.0806, 0.1764] | 0.0706 | 0.0695 |

## Lo robusto (independiente del prior)

- **La cota analítica de dominio** δ_H ≥ λ_H/√(b_max²−4·C0_lo·m_lo²) **supera 0.012 en todas las configuraciones** (verificado; mínimo muestreado ≥ cota en todas: sí). La inaccesibilidad del 0.012 es una propiedad de los dominios O(1), no del prior: cerrarlo exigiría b̄ ≳ 11 — fuera de cualquier lectura O(1). La regla canónica sigue demostrada desde dentro.
- **El orden de magnitud**: δ_H mediano entre 0.028 y 0.113 — pocas×10⁻² en todo el barrido.
- El filtro de fertilidad es marginal para δ_H (no depende de ē ni γR).

## Lo dependiente del prior (la afirmación queda condicionada)

- La mediana precisa varía hasta ~×4 con prior y dominio (0.028–0.113); el 0.054 de la Nota I es el valor de la configuración fiducial (uniforme, (0.5, 2), b̄ extendido).
- δ_H = 0.0581 cae dentro de la banda central [p5, p95] en 10/11 configuraciones; en la única SIN extensión de b̄ queda bajo el p5 (cola inferior): la elección del rango de b̄ importa y queda declarada como parte del muestreo, no de la naturaleza.

CONCLUSIÓN HONESTA: «ninguna forma O(1) cierra el empalme en 0.012» es robusto (analítico); «δ_H = pocas×10⁻² es genérico» es robusto (medido); «la mediana está en 0.054» es la lectura fiducial, no un invariante del paisaje.
