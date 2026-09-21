# La salida de S₀: bounce de Coleman, escape de Kramers y el reloj S arrancando en el estado nucleado (21-sep-2026)

Commit `c3aade7b9`. Formas del Basal por defecto; δ₀_max de metastabilidad = 0.1028.

**Lectura obligatoria**: bounce de Coleman sobre el corte radial del Basal completo (E8): dimensión d del instantón, prefactor A y normalización de la acción DECLARADOS; B(δ₀) medida; Γ₀(0) = 0 se cumple sin prefactor para d = 4 (B ∝ δ₀^{−1}) y exige A(δ₀) → 0 para d = 3.

## Bounce O(d) sobre el corte radial (paisaje completo)

| δ₀ | barrera/T₀ | d = 4: B | e^{−B} | f₀ | d = 3: B | e^{−B} | f₀ |
|---|---|---|---|---|---|---|---|
| 0.002 | 0.0809 | 6.708e+04 | 0.00e+00 | 0.9999 | 15.21 | 2.48e-07 | 0.9917 |
| 0.004 | 0.0710 | 2.936e+04 | 0.00e+00 | 0.9999 | 13.75 | 1.07e-06 | 0.9872 |
| 0.008 | 0.0584 | 1.213e+04 | 0.00e+00 | 0.9997 | 11.87 | 7.00e-06 | 0.9771 |
| 0.016 | 0.0429 | 4593 | 0.00e+00 | 0.9991 | 9.543 | 7.17e-05 | 0.9503 |
| 0.032 | 0.0252 | 1505 | 0.00e+00 | 0.9954 | 6.754 | 1.17e-03 | 0.8634 |
| 0.064 | 0.0077 | 357.8 | 3.98e-156 | 0.9486 | 3.443 | 3.20e-02 | 0.5219 |

Ley de escala medida: B ∝ δ₀^p con p = -1.486 (d = 4; argumento 3 − d = −1) y p = -0.403 (d = 3; argumento 0). Γ₀(0) = 0 sin prefactor para d = 4 (B → ∞); para d = 3 B tiende a una constante y el axioma exige A(δ₀) → 0. f₀ es la fracción de T₀ ya descargada en el centro del bounce: en este régimen (ε = T₀ ≫ altura de barrera, pared gruesa) el instantón entrega el campo casi en el vacío verdadero para d = 4 y parcialmente descargado para d = 3 — la descarga y sus umbrales caen DENTRO de la nucleación, no a lo largo del Flujo del Camino. La estimación de pared delgada no es aplicable (barrera ≪ ε) y se publica solo como referencia.

## Escape de Kramers del Flujo del Camino con difusión entrópica (Def. 4.4)

| δ₀ | ΔV_b | √(V''_fv|V''_b|)/2π | Γ_K (D = 1e-9) | Γ_K (1e-8) | Γ_K (1e-7) |
|---|---|---|---|---|---|
| 0.002 | 5.77e-10 | 7.98e-07 | 4.48e-07 | 7.53e-07 | 7.93e-07 |
| 0.004 | 4.15e-09 | 3.12e-06 | 4.93e-08 | 2.06e-06 | 2.99e-06 |
| 0.008 | 2.82e-08 | 1.20e-05 | 7.07e-18 | 7.20e-07 | 9.08e-06 |
| 0.016 | 1.73e-07 | 4.51e-05 | 4.84e-80 | 1.44e-12 | 8.02e-06 |
| 0.032 | 8.57e-07 | 1.59e-04 | 0.00e+00 | 1.01e-41 | 3.02e-08 |
| 0.064 | 2.24e-06 | 4.54e-04 | 0.00e+00 | 2.15e-101 | 8.41e-14 |

Exponentes medidos: prefactor ∝ δ₀^1.85 (argumento 2), barrera ∝ δ₀^2.44 (argumento 3): Γ_K(0) = 0 por el PREFACTOR (el paisaje plano no tiene curvatura que fije un ritmo), mecanismo opuesto al bounce. Cuál es la nucleación del tratado — instantón conservativo o escape disipativo de la dinámica de primer orden del Axioma 4 — es decisión del diccionario.

## El reloj S arrancando en el centro del bounce (δ₀ = 0.01, umbrales impuestos)

| d | f₀ | B | eventos en σ = 0 (dentro de la nucleación) | σ̂ del descenso restante | S ≡ f − f₀ |
|---|---|---|---|---|---|
| 3 | 0.9713 | 11.17 | ['colapso_1D', 'colapso_2D'] | 0.155 | 2.3e-13 |
| 4 | 0.9996 | 8974 | ['colapso_1D', 'colapso_2D', 'colapso_3D'] | 0.001 | 1.5e-17 |

## Integración conjunta Φ_Ad ⊗ λ_i desde el bounce (modo emergente, diagnóstico)

| d | τ | f₀ | S final | cruces D = 0 (S_emergent) | D mínimo | fin |
|---|---|---|---|---|---|---|
| 3 | 0.001 | 0.971 | 0.0251 | — | 4.94e-04 | max_steps agotado |
| 3 | 0.01 | 0.971 | 0.0124 | — | 4.71e-04 | max_steps agotado |
| 3 | 0.1 | 0.971 | 0.0023 | — | 4.48e-04 | max_steps agotado |
| 4 | 0.001 | 1.000 | 0.0000 | — | 5.00e-04 | descenso completado (residuo de descarga alcanzado) |
| 4 | 0.01 | 1.000 | 0.0000 | — | 5.00e-04 | descenso completado (residuo de descarga alcanzado) |
| 4 | 0.1 | 1.000 | 0.0000 | — | 5.00e-04 | descenso completado (residuo de descarga alcanzado) |

Ningún cruce D = 0 en el rango explorado: los eventos S_emergent se publican vacíos (sin estatuto; depende de τ).

## Declarado / lo que NO afirma

- dimensión d del instantón (3 y 4 publicadas; el tramo pre-geométrico no tiene espacio-tiempo)
- prefactor A de Γ₀ = A·e^{−B} (dimensional, no está en el tratado): se publica e^{−B}
- normalización de la acción euclidiana (G = 1, ħ = 1)
- D_ent de la difusión entrópica (el diccionario t ↔ σ de la Def. 4.4 no la fija)
- corte radial θ = 0 (salida al polo de masa, Prop. 3.5)

- ninguna de las dos nucleaciones es «la del tratado»: decidirlo es del diccionario
- ningún umbral emerge en la integración conjunta (los eventos S_emergent se publican vacíos si D no cruza cero)
- no es demostración física
