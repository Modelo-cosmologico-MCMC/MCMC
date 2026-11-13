# Decisión del autor del 13-nov-2026: apertura de la ronda 2 de R6 (cascada de habilitación)

Respuesta literal de Adrián Martínez Estellés a la pregunta «¿Abro una ronda 2?», hecha tras el INDETERMINADO de la ronda 1
(`results/2026-11-13_enabling_cascade`, preinscripción `9040df79900a…`). Las fórmulas van como el autor las escribió. El
acta de las decisiones 4, 5, 7 y 8 (`docs/decisiones_autor_2026-11-13.md`) no se toca: su sha256 está congelado en la
preinscripción de la ronda 1.

> R6 — Apertura de ronda 2 mediante integración logarítmica
> Decisión del autor: Sí, abrir la ronda 2 conservando íntegramente el modelo físico de la ronda 1.
>
> La primera ronda de R6 obtiene una letra global INDETERMINADO: el caso (\delta_0=0,003) recibe A, con cocientes 11,3 y 10,1 y controles superados; los casos (\delta_0=0,01) y (\delta_0=0,0554) no pudieron integrarse debido al subdesbordamiento numérico de las amplitudes de los modos.
>
> Considero necesario distinguir una pérdida de precisión de la representación numérica de una verdadera ausencia de activación dimensional.
>
> Por ello, autorizo una segunda ronda con las mismas ecuaciones físicas, condiciones iniciales, pesos, compuertas, controles, brazos, valores de (\delta_0) y criterios de aceptación.
>
> La única modificación admitida es sustituir la integración directa de las amplitudes positivas (q_k) por una formulación matemáticamente equivalente en (y_k=\ln q_k), incluyendo el tratamiento logarítmico de los acoplamientos y eventos que lo requieran.
>
> Antes de ejecutarla deberán verificarse la equivalencia de las ecuaciones, la conservación de los estados físicamente nulos y la convergencia numérica.
>
> La nueva preinscripción quedará congelada antes de realizar las corridas.
>
> Se declara expresamente que esta segunda ronda no es ciega, puesto que ya se conocen los resultados y cocientes de la primera. No se interpretará como una confirmación estadísticamente independiente.
>
> El brazo con fuente continua de ruido permanecerá como hipótesis física alternativa y no sustituirá el brazo principal.
>
> El objetivo de la ronda 2 es determinar si los casos anteriormente no integrables admiten una trayectoria numérica estable y si, bajo las reglas originales, se activan realmente los modos de estructuración de V1D, V2D y V3D.
>
> La letra final se asignará exclusivamente conforme a los criterios previamente congelados, conservando íntegramente los resultados negativos e indeterminados.

## Lo que esta decisión exige a la ronda 2

| Condición del autor | Cómo se cumple |
|---|---|
| Mismas ecuaciones, condiciones iniciales, pesos, compuertas, controles, brazos, δ₀ y criterios | La preinscripción de la ronda 2 lleva las reglas, los brazos y la rejilla de la ronda 1 sin tocar; un test lo vigila |
| Única modificación: integrar y_k = ln q_k | `amplitude_representation = "log"` en todos los brazos; d ln q_k/dσ = −r_k/G, con ∂V/∂q_k = r_k·q_k |
| Verificar antes la equivalencia de las ecuaciones | Tests sin inclinación: el gradiente latente es proporcional a la amplitud; la velocidad en logaritmo es la lineal dividida por q_k; los gatillos coinciden a 1e-6 |
| Verificar la conservación de los estados físicamente nulos | Tests: una dirección sin sembrar (ln q = −∞) sigue en cero exacto y, sin q_1, las superiores no se abren |
| Verificar la convergencia numérica | Test sin inclinación con el paso a la mitad; en la ronda, la puerta de convergencia congelada, celda a celda |
| Preinscripción congelada antes de correr | Commit propio, posterior al generador y anterior a las corridas |
| No es ciega ni confirmación independiente | Declarado en la preinscripción, con las cifras de la ronda 1 que ya se conocen |
| La fuente continua sigue siendo hipótesis alternativa | El brazo con suelo de ruido sigue como control; el brazo con letra es el de la semilla inicial |
| La letra, solo por los criterios ya congelados | El analizador lee las reglas de la preinscripción; la ronda 1 queda como salió |
