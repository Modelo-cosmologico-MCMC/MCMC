# Cascada de habilitación, ronda 1 — lectura a posteriori (sin letra)

Escrita después de ver los números, sobre las corridas de la preinscripción `9040df79900a…`. No cambia ninguna letra: la global de la ronda 1 es **INDETERMINADO** y así queda registrada. Aquí se dice por qué, con qué evidencia, y qué se puede leer de las celdas que sí terminaron.

## Por qué INDETERMINADO: el flotante agota la semilla de q_2

El brazo base integra las amplitudes q̂_k tal cual. Mientras su compuerta está cerrada (m_2² ≈ +1), q̂_2 decae como e^(−W_2·σ̂). En el juguete radial, con el que se comprobó el instrumento antes de congelar, W_2 = 336 y la semilla llega al gatillo en 10⁻²³⁰. En el reloj completo los pesos se fijan respecto de la Tensión Primordial MEDIDA, que la inclinación sube un 7 %, un 13 % y un 30 % en δ₀ = 0.003, 0.01 y 0.0554: W_2 = 361, 381 y 439, y además el Basal tarda más en descargarse. La amplitud llega entonces al gatillo por debajo de 4.9·10⁻³²⁴, el menor número que el flotante representa. Queda en un subnormal que ya no crece (un incremento del 13 % sobre una unidad del último dígito se redondea a cero), la dirección no se habilita y la corrida no termina.

La estimación de la tabla es log₁₀(semilla) − W_2·σ̂_gatillo/ln 10 (compuerta cerrada hasta el gatillo). En aritmética exacta la amplitud seguiría siendo no nula y la dirección se abriría.

| δ₀ | brazo | termina | colapsos | σ̂ del segundo gatillo | W_2 | semilla | log₁₀ de q̂_2 al abrirse | autovalor mínimo final |
|---|---|---|---|---|---|---|---|---|
| 0.003 | theta_0.95 | NO | 1 | 2.047 | 361.1 | 1e-06 | -327 | -270.1 |
| 0.003 | Delta_0.02 | sí | 3 | 1.591 | 361.1 | 1e-06 | -255 | 0.8 |
| 0.003 | Delta_0.1 | sí | 3 | 1.877 | 361.1 | 1e-06 | -300 | 0.6 |
| 0.003 | base | sí | 3 | 1.637 | 361.1 | 1e-06 | -263 | 0.8 |
| 0.003 | convergencia | sí | 3 | 1.637 | 361.1 | 1e-06 | -263 | 0.8 |
| 0.003 | pesos_iguales | sí | 3 | 5.485 | 3.6 | 1e-06 | -15 | 7.2 |
| 0.003 | referencia_ley | sí | 3 | 1.668 | 336.3 | 1e-06 | -250 | 0.7 |
| 0.003 | semilla_1e-3 | sí | 3 | 1.439 | 361.1 | 1e-03 | -229 | 0.8 |
| 0.003 | semilla_1e-9 | sí | 3 | 1.835 | 361.1 | 1e-09 | -297 | 0.8 |
| 0.003 | suelo_1e-18 | sí | 3 | 1.637 | 361.1 | 1e-06 | -18 | 0.8 |
| 0.003 | theta_0.8 | sí | 3 | 1.416 | 361.1 | 1e-06 | -228 | 0.8 |
| 0.01 | Delta_0.1 | NO | 1 | 2.008 | 381.2 | 1e-06 | -338 | -281.8 |
| 0.01 | semilla_1e-9 | NO | 1 | 1.942 | 381.2 | 1e-09 | -331 | -366.9 |
| 0.01 | theta_0.95 | NO | 1 | 2.176 | 381.2 | 1e-06 | -366 | -285.2 |
| 0.01 | Delta_0.02 | sí | 3 | 1.706 | 381.2 | 1e-06 | -288 | 0.8 |
| 0.01 | base | sí | 3 | 1.754 | 381.2 | 1e-06 | -296 | 0.8 |
| 0.01 | convergencia | sí | 3 | 1.754 | 381.2 | 1e-06 | -296 | 0.8 |
| 0.01 | pesos_iguales | sí | 3 | 5.399 | 3.8 | 1e-06 | -15 | 7.6 |
| 0.01 | referencia_ley | sí | 3 | 1.808 | 336.3 | 1e-06 | -270 | 0.7 |
| 0.01 | semilla_1e-3 | sí | 3 | 1.566 | 381.2 | 1e-03 | -262 | 0.8 |
| 0.01 | suelo_1e-18 | sí | 3 | 1.754 | 381.2 | 1e-06 | -18 | 0.8 |
| 0.01 | theta_0.8 | sí | 3 | 1.535 | 381.2 | 1e-06 | -260 | 0.8 |
| 0.0554 | Delta_0.02 | NO | 1 | 2.573 | 438.7 | 1e-06 | -496 | -438.7 |
| 0.0554 | Delta_0.1 | NO | 1 | 3.006 | 438.7 | 1e-06 | -579 | -325.1 |
| 0.0554 | base | NO | 1 | 2.636 | 438.7 | 1e-06 | -508 | -422.3 |
| 0.0554 | convergencia | NO | 1 | 2.636 | 438.7 | 1e-06 | -508 | -422.3 |
| 0.0554 | semilla_1e-3 | NO | 1 | 2.473 | 438.7 | 1e-03 | -474 | -422.3 |
| 0.0554 | semilla_1e-9 | NO | 1 | 2.800 | 438.7 | 1e-09 | -542 | -422.3 |
| 0.0554 | theta_0.8 | NO | 1 | 2.410 | 438.7 | 1e-06 | -465 | -438.4 |
| 0.0554 | theta_0.95 | NO | 1 | 3.168 | 438.7 | 1e-06 | -609 | -328.7 |
| 0.0554 | pesos_iguales | sí | 3 | 5.803 | 4.4 | 1e-06 | -17 | 8.7 |
| 0.0554 | referencia_ley | sí | 3 | 2.743 | 336.3 | 1e-06 | -407 | 0.7 |
| 0.0554 | suelo_1e-18 | sí | 3 | 2.164 | 438.7 | 1e-06 | -18 | 0.9 |

Sin suelo de ruido, las 17 celdas cuya amplitud estimada queda por encima del menor número representable (10^-323) terminan todas, la más baja con 10^-300. De las 13 que quedan por debajo, 12 no terminan y 1 sí (referencia_ley en δ₀ = 0.0554, con 10^-407): en el rango subnormal el resultado depende del redondeo y de la secuencia de pasos, que es otra forma de decir que ahí la representación lineal no es fiable. En todas las que no terminan el segundo gatillo sí se dispara (dos gatillos, un colapso) y el estado final es inestable en la dirección q_2 (autovalor mínimo ≈ W_2·m_2² < 0): la inestabilidad existe y lo que falta es una amplitud representable.

## Lo que se lee de las celdas que terminaron

- δ₀ = 0.003, brazo base: gatillos en S = 0.008662, 0.097837, 0.990064; cocientes 11.295 y 10.120; mayor ΔV/T_total = 9.9e-16.
- δ₀ = 0.01, brazo base: gatillos en S = 0.008675, 0.097847, 0.990064; cocientes 11.279 y 10.118; mayor ΔV/T_total = 9.4e-16.
- δ₀ = 0.003, con suelo de ruido de 1e-18 (fuente continua): 3 colapsos, gatillos en S = 0.008662, 0.097837, 0.990063; cocientes 11.295 y 10.120.
- δ₀ = 0.01, con suelo de ruido de 1e-18 (fuente continua): 3 colapsos, gatillos en S = 0.008675, 0.097847, 0.990064; cocientes 11.279 y 10.118.
- δ₀ = 0.0554, con suelo de ruido de 1e-18 (fuente continua): 3 colapsos, gatillos en S = 0.008706, 0.097871, 0.990064; cocientes 11.242 y 10.116.

Con una fuente continua la cascada se completa en los tres δ₀ y sus cocientes caen en [7, 14]. Es una lectura, no una letra: el brazo con letra es el de la semilla inicial, y en dos de los tres δ₀ no se pudo integrar.

## Dos avisos sobre la tabla de sensibilidad del análisis

- En las celdas de sensibilidad con un solo colapso (θ_g = 0.95; Δ_g = 0.1 en δ₀ = 0.01; las cuatro de la compuerta en δ₀ = 0.0554) la corrida no terminó por el mismo motivo. La columna «lectura de la regla» pone C porque la regla ve menos de tres colapsos; no es un resultado sobre la compuerta. El JSON del análisis lleva `finished` en cada fila; la tabla, no.
- En δ₀ = 0.0554 los controles «semilla_1e-3» y «semilla_1e-9» figuran como superados: tienen el mismo único colapso que el brazo base, que tampoco terminó. Es lo que dice la regla congelada, y no significa que los umbrales se hayan comprobado ahí.

## Lo que cambia una ronda 2, si el autor la abre

Integrar ln q̂_k en lugar de q̂_k. Todo el gradiente latente es proporcional a la amplitud, de modo que d ln q̂_k/dσ̂ es regular aunque q̂_k sea diminuta, y la semilla no se agota. La opción está en `core/enabling_cascade.py` (`amplitude_representation="log"`) y en el juguete radial reproduce los gatillos de la representación lineal a siete cifras, también con semilla 1e-30. Las reglas, las puertas, los controles y los brazos serían los mismos. Las cifras de arriba ya se conocen: la ronda 2 no sería ciega para las celdas que terminaron en la ronda 1.

