---
title: "Modelo Cosmológico de Múltiples Colapsos — Tratado de Fundamentos"
subtitle: "Lista de erratas para la v36 (lista de trabajo, 12 de noviembre de 2026)"
date: "12 de noviembre de 2026"
lang: es
---

> **Qué es este documento.** Una lista de trabajo para la v36: las erratas
> del Tratado de fundamentos v35 que la revisión profunda del 10 y 11 de
> noviembre de 2026 dejó señaladas y que no requieren ninguna decisión del
> autor. Es el PR R5 del plan de acción del 11-nov.
>
> **Qué no es.** No es una fe de erratas: no reclasifica ningún resultado
> ni sustituye a la FE-08-2026
> (`docs/erratas_epistemologicas_2026-08-04.md`), que sigue vigente. Nada
> de lo que sigue es canon hasta que el autor lo adopte en la v36. Tampoco
> es la lista completa de erratas del corpus (véase la sección 4).

**Cómo leer el estatuto de cada entrada.** «Comprobada en el repositorio»
quiere decir que hay una función y un test que reproducen la cuenta; es
una comprobación interna (E8), nunca una demostración física. «Señalada
por la revisión» quiere decir que la cuenta es de los informes del 5, 10 u
11 de noviembre y que aquí solo se comprueba la parte que el repositorio
puede comprobar: el texto del Tratado no está en el repositorio. «Pendiente
de confirmar la cita» quiere decir que el álgebra es correcta, pero no se
ha podido comprobar que el texto citado diga eso. La columna «Qué dice la
v35» resume cada enunciado tal como lo recogen el código del repositorio y
los informes de la revisión: antes de adoptar una entrada hay que cotejarla
con el texto del Tratado.

# 1. Erratas que no requieren decisión

| N.º | Enunciado | Qué dice la v35 | Qué se propone | Comprobación |
|---|---|---|---|---|
| 1 | Prop. 3.4, residuo de desequilibrio | ⟨χ⟩ ≃ (ē/m̄²)·δ₀ en el falso vacío | Dentro del dominio físico C4, ⟨χ⟩ ≃ (ē/2m̄²)·δ₀. La fórmula actual es el mínimo sin restringir, que cae en φ_E < 0 | Comprobada en el repositorio |
| 2 | Prop. 3.4, inclinación | La inclinación es subdominante, O(δ₀^(7/2)) | Añadir que, relativa a T₀, es O(δ₀^(1/2)) con coeficiente κ₁ = ē√κ₊/(√2·c̄). No es pequeña en los δ₀ que usa el programa | Comprobada en el repositorio |
| 3 | Teo. 4.5, Monotonía | V baja, S sube y T baja a lo largo del Camino | Añadir al enunciado la hipótesis «con acoplos fijos» | Comprobada en el repositorio |
| 4 | Lema 7.3, Precedencia | El sellado es condición necesaria | Separar dos papeles: suficiente para la positividad, y necesario solo para un hamiltoniano independiente de S | La suficiencia, comprobada en el repositorio (juguete); el segundo papel, señalado por la revisión |
| 5 | Apéndice B.3, balance de canales | Las ecuaciones de refresco de ρ_lat y ρ_id | Decir que el balance no conserva el total y remitir a la forma cerrada del programa | Comprobada en el repositorio |

**1. Residuo de la Prop. 3.4.** Junto al origen V ≃ ½M₀²ρ² − η·χ. Su
mínimo sin restringir está en φ_M = −φ_E = η/(√2·M₀²), fuera del dominio
físico C4 (el primer cuadrante del Plano Dual). Dentro del dominio el
falso vacío queda sobre la frontera φ_E = 0 y el residuo es la mitad.
Cociente medido/C4: 1.005 / 1.016 / 1.113 en δ₀ = 0.003 / 0.01 / 0.0554.
El mínimo sin restringir solo existe por debajo de δ₀ = 0.0514: en δ_H =
0.0554 no existe. Fila `residuo-falso-vacio-c4`;
`core.basal.chi_residue_c4`; `tests/test_chi_residue_c4.py`.

**2. Inclinación de la Prop. 3.4.** η·ρ₊ = O(δ₀^(7/2)) es correcto en valor
absoluto. Frente a T₀ = c̄·δ₀³ la corrección es κ₁·√δ₀ con κ₁ = 1.361 para
las formas por defecto: +14.6 % en δ₀ = 0.012, +30.4 % en δ_H = 0.0554 y
+31.1 % en δ₀ = 0.058. El reloj S ya mide T₀ en el paisaje completo y
publica la corrección (`core.s_clock.kappa1_tilt`, comprobación
`T0_tilt_first_order`).

**3. Hipótesis del Teo. 4.5.** La prueba usa dV/dσ = −(∇V)ᵀG⁻¹∇V, que vale
con los acoplos fijos. Con acoplos en movimiento V no es función de
Lyapunov: el reloj S ya lo marca en su modo emergente. La cuenta es de la
fila `cruce-victoria-retorno`: llevar el estado ocupado hasta la espinodal
D = 0 sube su energía en más de una Tensión Primordial (1.198 / 1.669 /
1.132·T₀ por los tres caminos de un solo acoplo). `core.decade`;
`tests/test_victoria_return.py`.

**4. Dos papeles del Lema 7.3.** Para la positividad por reflexión basta
la simetría especular de los acoplos respecto de la loncha: el sellado es
suficiente, no necesario (fila `rp-no-estacionaria`, en el juguete
escalar; la necesidad no está demostrada). Para la reconstrucción con un
hamiltoniano independiente de S (Cor. 7.2) hace falta además la
invariancia bajo traslaciones en S, que solo existe con los acoplos
sellados: ese es el papel en que el sellado sí es necesario, y lo señala
la revisión como resultado conocido, no un test del repositorio. En el
Camino con difusión la condición es estacionariedad con balance detallado
(fila `rp-camino-onsager-machlup`, en una dimensión).

**5. Balance del Apéndice B.3.** El refresco de canales es ρ_lat ← ρ_lat +
Δt·(−κ_lat·ρ_lat + Γ_lat) y ρ_id ← ρ_id + Δt·(+η_dir·ρ_m + Γ_act), con
ρ_m sin tocar (`cronos.channels.refresh_channels`). El total cambia en
Δt·(−κ_lat·ρ_lat + Γ_lat + η_dir·ρ_m + Γ_act), que no se anula salvo que
las fuentes se elijan para ello: ninguna ecuación lo impone. La revisión
encontró tres versiones del mismo balance con los símbolos en papeles
opuestos (octubre de 2025 y Ap. B.3: κ_lat es una tasa de decaimiento;
Maestro V2, ec. (295): κ_lat es una tasa de sellado), y que la ec. (299)
del Maestro remite a (200)–(203), que no son las suyas. El programa
escribe hoy el trasvase con una sola Q, que sale de la ECV y entra en la
MCV: ρ̇_M + 3Hρ_M = Q y ρ̇_E = −Q, que cierra por construcción (fila
`brazo-dos-sentidos-todas-las-posibilidades`, sobre `mcv-equilibrio-c` e
`intercambio-masa-borde-ecv`). Qué ley Q se adopta es la decisión 8 del
autor; que el Ap. B.3 no cierra no depende de ella.

# 2. Pendientes del informe del 5 de noviembre

| N.º | Enunciado | Qué dice la v35 | Qué se propone | Comprobación |
|---|---|---|---|---|
| 6 | Prop. 10.2, Modo de Memoria | m_θ² = η/ρ₊ en el anillo de vacíos | Decir que esa curvatura está en θ = −π/4, fuera de C4. Dentro de C4, y sin término angular, el vacío está sobre la frontera θ = 0, con pendiente η/√2 | Comprobada en el repositorio |
| 7 | (12.1), (H.8) y F.3: el acoplo β₃ | El mismo símbolo vale 0.13 en (12.1) y 0.0325 en (H.8) y F.3 | Una sola normalización, o dos símbolos | Comprobada en el repositorio |
| 8 | A.6, ecuación de estado | «w de ∼0 a alto z a −1 hoy» | Aclarar a qué canal se refiere. La Λ_rel de A.3 tiene w ≤ −1 en todo z | Señalada por la revisión; la cuenta de Λ_rel, comprobada |

**6. Modo de Memoria.** Sobre el anillo ρ = ρ₊ el Basal vale V = cte −
η·ρ₊·cos(θ + π/4): su mínimo angular está en θ = −π/4, donde φ_E < 0, y
allí la curvatura da m_θ² = η/ρ₊ ∝ δ₀^(5/2) (6.18·10⁻⁶ con δ₀ = 0.01 y las
formas por defecto; `core.victoria.memory_mode_mass_sq`). El exponente 5/2
es correcto. En θ = 0, la frontera de C4, lo que hay es una pendiente η/√2
por unidad de arco, que empuja hacia fuera del dominio: una pared lineal,
no un pozo. Con el término angular del 23-oct el modo χ sí tiene masa μ²
dentro de C4 (fila `cuarta-ecuacion-termino-angular`); que ese término
entre en el canon depende de la decisión 2.

**7. Normalización de β₃.** m_H = √(2β₃)·v en (12.1) exige β₃ = λ_H = 0.13;
m_H² = 8β₃v² en (H.8) exige β₃ = λ_H/4 = 0.0325. El factor 4 es el que
separa las formas (Φ² − v²)² y Φ⁴. El repositorio ya expone los dos
convenios por separado (`mass_program/B7_empalme.py`, el de (12.1) es el
canónico aquí). Si son dos acoplos distintos, la errata es solo de
notación (suposición S3 del informe del 5-nov).

**8. Frase de A.6.** Con Λ_rel(z) ∝ 1 + ε_Λ·tanh((z_trans − z)/Δz) y los
valores del canon (ε_Λ = 0.012, z_trans = 8.9, Δz = 1.5), w = −1 +
(1/3)·d ln Λ_rel/d ln(1+z) es ≤ −1 en todo z: mínimo −1.027 junto a
z_trans y w(0) + 1 = −7·10⁻⁸ (`cosmology.background.Lambda_rel`). El
repositorio lee la frase de A.6 como la del canal indeterminado, w_id → 0
a z alto y → −1 a z bajo (`cosmology/dark_channels.py`), que es otro
objeto. Si la frase se refiere a Λ_rel, contradice a A.3; si se refiere a
w_id, basta decirlo. Lo decide el texto, que no está en el repositorio.

# 3. Tratados de 2025: pendiente de confirmar la cita

El documento «Recuperación de la ontología original a partir de los
tratados antiguos» (11-nov) atribuye cinco fórmulas a los tratados del
24-mar-2025 y del 10-oct-2025. La revisión rehízo las cinco cuentas y son
correctas como álgebra. Esos dos PDF no están en el Proyecto ni en el
repositorio, así que no se ha comprobado que digan literalmente lo que se
les atribuye: **ninguna de las cinco es una errata hasta que el autor
confirme la cita.**

| N.º | Fórmula atribuida | La cuenta | Estatuto |
|---|---|---|---|
| 9 | t_rel(S) = (λ/kα)·ln cosh(S/λ) con derivada ∝ λ | d/dS = tanh(S/λ)/(kα): el factor es 1/(kα), sin λ. Con λ = 10⁻² la diferencia es un factor 100 | Pendiente de confirmar la cita |
| 10 | Mínimos de V = −μ²Φ² + λΦ⁴ | Están en Φ = ±μ/√(2λ), con profundidad −μ⁴/(4λ) | Pendiente de confirmar la cita |
| 11 | η = ε·exp(ε·N) con ε_CTP = 0.001/0.099 = 0.0101 | Es creciente en N: η ≥ ε_CTP para N ≥ 0. No puede dar 6·10⁻¹⁰ | Pendiente de confirmar la cita |
| 12 | Balance de tres sectores | Las tres ecuaciones suman (η_dir − δ_dir)·ρ_m − 2κ_lat·ρ_lat: no cierra salvo η_dir = δ_dir y κ_lat = 0 | Pendiente de confirmar la cita; el Ap. B.3 es la entrada 5 |
| 13 | g = ḡ·(1 + Φ_ten/M_Pl) «degenera» en Φ_ten = 0 | En Φ_ten = 0 da g = ḡ: no degenera | Pendiente de confirmar la cita |

Las cuentas 9, 10, 11 y 13 se repiten en `tests/test_erratas_v36.py` sin
cálculo simbólico. La 12 es de la revisión (bloque A de su guion): las
tres ecuaciones no están transcritas en el repositorio.

# 4. Lo que esta lista no cubre

**Esperan una decisión del autor**, así que no son erratas todavía:

- **Prop. 3.5 y Def. 2.2** (salida por el polo de masa y cruce de la
  diagonal). Con el término angular ½μ²χ² las dos mitades se excluyen
  (fila `cuarta-ecuacion-collado`). Qué mitad se conserva es la decisión 2.
- **Obs. 8.6 y Prop. 10.7** (el Cruce de Victoria como gatillo de la
  cascada). Con un solo Potencial Basal, D = 0 es un retorno (fila
  `cruce-victoria-retorno`). Dónde vive el Cruce es la decisión 1.
- **Teo. 7.1 y §13.3** (sobre qué acción se enuncia la positividad). La
  del propio Camino es la de Onsager–Machlup (fila
  `rp-camino-onsager-machlup`). Es la decisión 3.

**Señaladas en los informes del 5-nov y del 27-oct y no desarrolladas
aquí**: la numeración de los sellos en (5.1), F.2 y la Década; la cifra
de la cota (11.5), donde «∼10⁻⁶» corresponde a v ≈ 300 km/s y para v =
10–200 km/s (v/c)² va de 1.1·10⁻⁹ a 4.5·10⁻⁷; el contraste BBN de H.2.5;
los restos de versiones anteriores en A.2, B.1–B.3, I.5, A.7 y F.3–F.4; y
las nueve erratas del repaso del Maestro V2 y del Unificado del 27-oct.
Su detalle está en esos informes, no en el repositorio.

# 5. Procedencia y comprobación

| Entradas | Procedencia | Dónde se comprueba |
|---|---|---|
| 1, 3 | Revisión profunda del 10-nov, hallazgos 4 y 1 | `tests/test_chi_residue_c4.py`, `tests/test_victoria_return.py` |
| 2, 6, 7, 8 | Informe del 5-nov | `tests/test_erratas_v36.py` |
| 4 | Informe del 5-nov y revisión del 10-nov, hallazgo 3 | `tests/test_rp_nonstationary.py`, `tests/test_camino_onsager_machlup.py` |
| 5, 12 | Continuación de la revisión, 11-nov | `tests/test_erratas_v36.py` (entrada 5) |
| 9, 10, 11, 13 | Continuación de la revisión, 11-nov | `tests/test_erratas_v36.py` (solo el álgebra) |

Las cifras de este documento salen de las formas O(1) por defecto del
repositorio (m̄ = 1, b̄ = 3, C₀ = 1, ē = 1). Los informes del 5-nov usaron
b̄ = 2.5 en algún guion: las formas cerradas no cambian y los coeficientes
sí (por ejemplo, η/ρ₊ = 7.07·10⁻⁶ allí y 6.18·10⁻⁶ aquí).
