# Cuarta ecuación, brazo «término angular»: el collado de salida de S₀ y el umbral A_c, sin letra

Commit `fcc8cdb8c`. V = V₀ + ½μ²χ², μ² = 4A(c̄/κ₊)δ₀² (ClockConfig.angular_A); puntos críticos interiores en R(ρ²) = 0 con χ = η/μ².

## Formas cerradas con A = 1 (las cifras del 23-oct)

| δ₀ | t | A_c | ángulo del collado | del collado a la diagonal | autovalores del hessiano en el collado (÷δ₀²) | barrera del polo / barrera del collado | θ* |
|---|---|---|---|---|---|---|---|
| 0.003 | 0.1054 | 0.0976 | 0.7164 | 0.0690 | -1.705, +1.281 | 2.580 | 0.7590 |
| 0.01 | 0.1924 | 0.1781 | 0.6591 | 0.1263 | -1.697, +1.273 | 2.343 | 0.7373 |
| 0.0554 | 0.4529 | 0.4192 | 0.4844 | 0.3010 | -1.642, +1.219 | 1.778 | 0.6719 |

Con A = 1 la barrera del rayo del polo no es un collado en ninguna de las tres: allí ∂V/∂φ_E < 0 y el campo baja hacia el interior del cuadrante.

## Corrida de consistencia: el reloj desde el rayo del collado frente al rayo declarado (A = 1)

| δ₀ | θ inicial (collado) | θ final | θ* | rotación que queda desde el collado | fracción de la del polo | θ final desde θ = 0 | θ monótona | S ≡ f y Monotonía |
|---|---|---|---|---|---|---|---|---|
| 0.003 | 0.7164 | 0.7397 | 0.7590 | 0.0427 | 5.6% | 0.7367 | sí | sí |
| 0.01 | 0.6591 | 0.7148 | 0.7373 | 0.0782 | 10.6% | 0.7148 | sí | sí |
| 0.0554 | 0.4844 | 0.6491 | 0.6719 | 0.1875 | 27.9% | 0.6492 | sí | sí |

Las dos corridas terminan en el residuo de descarga (f = 1 − ΔS), antes de que θ alcance θ*: el último tramo de la rotación es lento. Desde el collado, la «rotación continua desde el polo» del 23-oct se reduce a su último 6%–28%.

## Salida por el polo con A = 0.8·A_c

| δ₀ | A | collado interior | la barrera del polo es un collado | del vacío a la diagonal (forma cerrada) | donde acaba el descenso | θ monótona |
|---|---|---|---|---|---|---|
| 0.003 | 0.0780 | no | sí | 0.3444 | 0.3445 | sí |
| 0.01 | 0.1425 | no | sí | 0.3444 | 0.3444 | sí |
| 0.0554 | 0.3354 | no | sí | 0.3444 | 0.3444 | sí |

Con salida por el polo el vacío queda a arcsin√(κ₋/2κ₊) = 0.2735 rad (15.7°) o más de la diagonal, sea cual sea δ₀.

## Acercar el vacío a la diagonal aleja el collado del polo

sin(d_collado) = √(κ₊/κ₋)·sin(d_vacío), sin δ₀, A ni ē:

| del vacío a la diagonal (rad) | del collado a la diagonal (rad) | ángulo del collado desde el polo (rad) |
|---|---|---|
| 0.2 | 0.5470 | 0.2384 |
| 0.1 | 0.2644 | 0.5210 |
| 0.05 | 0.1312 | 0.6542 |

## Comprobaciones «derivadas» del reloj que estos brazos no superan

Están escritas para el rayo del polo sin término angular; se publican, no se ocultan:

- `S0/T0_tilt_first_order`: la corrección de la inclinación a T₀ a primer orden es la del Basal sin término angular.
- `S0/delta0_max_closed_form`: la bisección de δ₀_max busca hasta δ₀ = 2 y junto a la diagonal el máximo es mayor.
- `descent/exit_to_mass_pole_3_5`: exige θ final < 0.35: con el término angular el campo no se queda en el polo.

Cruza la diagonal en alguna corrida: **no**; identidades del descenso en todas: **sí**.

**Lectura**: sin letra (E8, E13): con ½μ²χ², salir por el polo de masa y llegar junto a la diagonal se excluyen. Con A = 1 el collado más bajo ya está cerca de la diagonal y la barrera del rayo del polo es de 1.8 a 2.6 veces más alta; con A ≤ A_c la salida es por el polo y el vacío queda a 15.7° o más de la diagonal. Las cifras del 23-oct valen como descenso desde el rayo θ = 0, que es la forma declarada del reloj. Qué mitad de la Prop. 3.5 se conserva es la decisión 2 del autor.
