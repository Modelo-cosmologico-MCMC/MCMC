# Ronda 3 del halo — declaración previa a la segunda ejecución (18-oct-2026)

Esta nota se escribe y se publica **antes** de volver a ejecutar las corridas. No modifica la preinscripción
congelada: `preregistration.json` sigue siendo byte a byte el del commit `cd4f4a4` (sha256
`6fba6b53d92910db86e21ed5ae9c8943d9c50f32dcb33b52e3d6eae01bf513ec`, comprobado hoy) y ninguna regla, puerta o
letra se toca.

## 1. Qué pasó con la primera ejecución

Las 17 corridas se lanzaron el 30-sep a las 14:27 UTC bajo esa preinscripción, en el commit `cd4f4a4`, con cuatro
trabajadores. En la noche del 30-sep catorce brazos habían terminado, dos seguían en curso y uno no había empezado.
Ningún fichero de corrida se había confirmado en el repositorio. El contenedor de cómputo se recicló por inactividad y **todos los ficheros de corrida y
los puntos de control se perdieron**.

La sesión de código vio números parciales en los registros de progreso de esa primera ejecución. No se publican: no
queda ningún fichero que los respalde. Tampoco se usan para nada. La segunda ejecución usa el mismo commit, la misma
preinscripción y las mismas reglas. Las corridas que agotan el presupuesto de pared de 4 h pueden llegar a un t final
algo distinto al de la primera ejecución, porque ese presupuesto se mide en tiempo de reloj; se publica tal cual.

## 2. Dos defectos de la regla congelada, demostrables sin corridas

**D1 — la letra A es inalcanzable por construcción.** La regla A exige que el brazo saturante «above» salga del
régimen débil. En el código, salir significa que ε_c máx supere `weak_regime_eps_max` = 1e-3. La forma saturante
acota ε_c < ε_max, y el punto «above» tiene ε_max = 1.62e-8 (1.5 × ε_max_crit). Por tanto «above» no puede
salir nunca y A no puede darse. Es un error del generador. La predicción del 28-sep («el segundo sale») se refería a
la inestabilidad de Cronos–Jeans (q > 1 en la cúspide), pero la prueba codificada es el umbral del régimen débil.

**D2 — una condición de A y de B no estaba cualificada.** A y B exigen que los cuatro brazos del barrido a
0.05·A_Sculptor salgan del régimen débil. La cualificación de capas, publicada antes de congelar, no vio ninguna
salida a 0.05·A_Sculptor en ninguna de sus celdas (ε_c máx ≤ 4.6e-4 con 480 s de pared). La ronda 3 da 4 h por
brazo, así que la condición no es imposible, pero ninguna cualificación la mostró alcanzable.

**Consecuencia bajo la regla congelada.** Los desenlaces alcanzables son C, B (solo si salen los cuatro brazos a
0.05·A_Sculptor y también el saturante «sub») e INDETERMINADO. Se ejecuta y se publica lo que la regla dé. Corregir
D1 y D2 exige una preinscripción nueva (ronda 4), no esta.

## 3. La revisión del autor del 30-sep

La revisión del autor pide puertas de energía por celda (3 × el suelo de la misma celda N, ds, ε_soft y rango),
evaluadas en el instante de salida t_weak, presupuestos de pared declarados por celda y estacionariedad por celda.
Llegó después de la congelación y no se aplica a esta ronda. Se implementa como regla del programa para las
preinscripciones futuras, en un PR aparte.

El análisis de esta ronda publicará además, **sin letra**, una tabla que pone |ΔE/E| de cada brazo junto al suelo
de la celda equivalente de la cualificación, donde esa celda exista.
