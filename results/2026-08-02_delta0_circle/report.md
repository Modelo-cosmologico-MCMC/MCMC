# El círculo de δ₀ — ronda 5, prioridad nº 1

La pregunta: ¿converge el mapa de retorno de Victoria (Teo. 10.6, con Techo, en la región fértil) al δ_H ≈ 0.0581 que el empalme C¹ midió sin m_H como entrada (H.2.4, B7)?

## Lo que el cálculo establece

Con el mapa lineal + Techo de H.2.3, en TODA la región fértil (A > 1) el atractor es el Techo, independiente de γR y del δ inicial:

    δ∞ = δ_sat = (W_max/c̄)^{1/3}

Medido iterando `core/victoria.py`: desde δ = 0.012 con γR = 1.5 y W_max = c̄·δ_H³, el retorno llega a δ∞ = 0.05814 en 4 vueltas (δ_H = 0.05814). Control negativo: con W_max/2 el atractor cae a 0.04614 — el 0.058 NO es intrínseco al mapa.

## El desenlace (publicable tal cual)

**El círculo se cierra ⟺ W_max = c̄·δ_H³ = 1.6523e-04** (formas fiduciales, c̄ = 0.8408). Es una ECUACIÓN DE CONSISTENCIA que liga el Techo de Victoria (Lema 10.3 / Teo. 10.6) con el empalme C¹ (ec. H.8). El tratado declara el Techo pero no asigna valor numérico a W_max, de modo que este cálculo no cierra ni rompe el círculo: TRANSFIERE la pregunta a W_max, que espera la microdinámica del reinicio (frente abierto nº 4). Ambos desenlaces (cierre o Silencio) siguen abiertos y ambos serían publicables.

Estatuto: condicional (F.2): el círculo se cierra ⟺ W_max = c̄·δ_H³; el tratado declara el Techo (Lema 10.3) pero no asigna valor a W_max — la ecuación de consistencia liga Teo. 10.6 con H.8 y transfiere la pregunta a W_max.

## Factores de consistencia si δ₀ = δ_H

Frente al 0.012 de ε_Λ que el v32 identificaba con δ₀ (identificación superada — regla canónica):

| Magnitud | Ley | Factor |
|---|---|---|
| T₀ (Prop. 3.4) | ×(δ_H/0.012)³ | ×113.7 |
| m_θ² (ec. 10.1) | ×(δ_H/0.012)^{5/2} | ×51.7 |

Estas dos escalas son las consecuencias a rastrear en el tratado si el círculo se cerrara en δ_H.

## Fertilidad re-centrada: la contención por paisaje

La condición de fertilidad ν > 0 ⟺ A > 1 (H.7) NO depende de δ₀ — el barrido del frente nº 4 no cambia. Lo que sí varía por paisaje es la CONTENCIÓN: W_max ≥ c̄(formas)·δ_H(formas)³.

Monte Carlo (semilla 20260802, formas O(1) en (0.5, 2.0), γR ~ U(0, 3.0]): 28782 paisajes fértiles (fracción 0.720).

- δ_H: mediana 0.0544, [5%, 95%] = [0.0367, 0.1155], mínimo muestreado 0.0332 — la alternativa «formas no fiduciales» del desenlace de B7, cuantificada: en TODA la región O(1) muestreada δ_H = O(0.05); el discriminante máximo alcanzable acota δ_H ≳ 0.033, así que NINGUNA forma O(1) cierra el empalme en δ₀ = 0.012 — el 0.012 de ε_Λ queda fuera del rango alcanzable, en refuerzo independiente de la regla canónica.
- W_max requerido: mediana 1.323e-04, [5%, 95%] = [3.330e-05, 5.704e-04].

Fracción de paisajes fértiles CONTENIDOS según W_max:

| W_max | fracción contenida |
|---|---|
| 1.652e-05 | 0.019 |
| 8.262e-05 | 0.325 |
| 1.652e-04 | 0.576 |
| 3.305e-04 | 0.803 |
| 1.652e-03 | 1.000 |

Parámetros declarados del cálculo: suelo δ_min = 0.0001 (Teo. 10.6 lo exige > 0 pero no lo cuantifica); γ_max = 3.0 (cartografía, no derivación).
