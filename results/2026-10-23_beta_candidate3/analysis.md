# Candidato 3 «sentido del flujo» — sin escalón: **C**; con escalón: **C**

Preinscripción `0eb5b7b6036c…`; commit `3c50b265f`. Letra por δ₀; la global de cada sub-brazo es la peor.

| δ₀ | escalón | cruces en s = S·abs(τ) | cocientes | cuártico | letra | motivo |
|---|---|---|---|---|---|---|
| 0.003 | no | 0.1871 |  |  | C | 1 cruces (< 3) |
| 0.003 | sí | 0.1871, 0.2642, 0.3051 | 1.41, 1.15 | 1.47, 1.19 | C | algún cociente fuera de [3, 30] |
| 0.01 | no | 0.1853 |  |  | C | 1 cruces (< 3) |
| 0.01 | sí | 0.1853, 0.2625, 0.3033 | 1.42, 1.16 | 1.47, 1.19 | C | algún cociente fuera de [3, 30] |
| 0.0554 | no | 0.1743 |  |  | C | 1 cruces (< 3) |
| 0.0554 | sí | 0.1743, 0.2512, 0.2921 | 1.44, 1.16 | 1.51, 1.2 | C | algún cociente fuera de [3, 30] |

## Controles hacia el IR (sin letra)

- δ₀ = 0.003, escalón no: cruces [] (acoplos fuera del dominio (C₀ ≤ 0))
- δ₀ = 0.003, escalón sí: cruces [] (acoplos fuera del dominio (C₀ ≤ 0))
- δ₀ = 0.01, escalón no: cruces [] (acoplos fuera del dominio (C₀ ≤ 0))
- δ₀ = 0.01, escalón sí: cruces [] (acoplos fuera del dominio (C₀ ≤ 0))
- δ₀ = 0.0554, escalón no: cruces [] (acoplos fuera del dominio (C₀ ≤ 0))
- δ₀ = 0.0554, escalón sí: cruces [] (acoplos fuera del dominio (C₀ ≤ 0))

## Reloj completo (complemento, sin letra)

- δ₀ = 0.003, τ = 0.01, escalón no: S ≥ S_end (fin del recorrido); cruces S = []; inestabilidades de masa 0
- δ₀ = 0.003, τ = 0.1, escalón no: S ≥ S_end (fin del recorrido); cruces S = []; inestabilidades de masa 0
- δ₀ = 0.003, τ = 1, escalón no: max_steps agotado; cruces S = []; inestabilidades de masa 0
- δ₀ = 0.003, τ = 0.01, escalón sí: S ≥ S_end (fin del recorrido); cruces S = []; inestabilidades de masa 0
- δ₀ = 0.003, τ = 0.1, escalón sí: S ≥ S_end (fin del recorrido); cruces S = []; inestabilidades de masa 0
- δ₀ = 0.003, τ = 1, escalón sí: max_steps agotado; cruces S = []; inestabilidades de masa 0
- δ₀ = 0.01, τ = 0.01, escalón no: S ≥ S_end (fin del recorrido); cruces S = []; inestabilidades de masa 0
- δ₀ = 0.01, τ = 0.1, escalón no: S ≥ S_end (fin del recorrido); cruces S = []; inestabilidades de masa 0
- δ₀ = 0.01, τ = 1, escalón no: max_steps agotado; cruces S = []; inestabilidades de masa 0
- δ₀ = 0.01, τ = 0.01, escalón sí: S ≥ S_end (fin del recorrido); cruces S = []; inestabilidades de masa 0
- δ₀ = 0.01, τ = 0.1, escalón sí: S ≥ S_end (fin del recorrido); cruces S = []; inestabilidades de masa 0
- δ₀ = 0.01, τ = 1, escalón sí: max_steps agotado; cruces S = []; inestabilidades de masa 0
- δ₀ = 0.0554, τ = 0.01, escalón no: S ≥ S_end (fin del recorrido); cruces S = []; inestabilidades de masa 0
- δ₀ = 0.0554, τ = 0.1, escalón no: S ≥ S_end (fin del recorrido); cruces S = []; inestabilidades de masa 0
- δ₀ = 0.0554, τ = 1, escalón no: descenso detenido en la frontera (velocidad proyectada ≈ 0; S no avanza); cruces S = [0.17434]; inestabilidades de masa 0
- δ₀ = 0.0554, τ = 0.01, escalón sí: S ≥ S_end (fin del recorrido); cruces S = []; inestabilidades de masa 0
- δ₀ = 0.0554, τ = 0.1, escalón sí: S ≥ S_end (fin del recorrido); cruces S = []; inestabilidades de masa 0
- δ₀ = 0.0554, τ = 1, escalón sí: S ≥ S_end (fin del recorrido); cruces S = [0.17434, 624.836668]; inestabilidades de masa 0

## Criterios de PR-6 (sin letra)

- (i), forma integral exacta (R llega a 1 antes de M₀² = 0, es decir, primer cruce con M₀² > 0): δ₀ = 0.003/no: True, δ₀ = 0.003/sí: True, δ₀ = 0.01/no: True, δ₀ = 0.01/sí: True, δ₀ = 0.0554/no: True, δ₀ = 0.0554/sí: True.
- (i), forma original (D monótono hasta el primer cruce): 0.003/no: False, 0.003/sí: False, 0.01/no: False, 0.01/sí: False, 0.0554/no: False, 0.0554/sí: False. Hacia el UV, D = 4C₀M₀²(R − 1) crece al principio aunque R decrezca, porque C₀M₀² crece.
- (iii): con el τ que pondría el primer cruce en S = 0.009, los otros dos caen en 0.009, 0.01271, 0.01468; 0.009, 0.01274, 0.01473; 0.009, 0.01297, 0.01508 (la Década pediría 0.099 y 0.999).

Nota del reloj completo: con τ = 1 y escalón, el segundo «cruce» en S = 624.8 cae en un único paso tras la reposición de M₀² (el paisaje cambia tanto que dS de un paso supera el fin del recorrido): límite del instrumento, sin letra. Con τ ≤ 0.1 el reloj termina en S = 1.001 antes de s = S·τ ≈ 0.17–0.19; con τ = 1 y δ₀ ≤ 0.01 agota max_steps antes.

**Lectura**: cribado diagnóstico (E8, E13): ninguna letra decide λ ni el sentido del flujo (decisión 1 del autor).
