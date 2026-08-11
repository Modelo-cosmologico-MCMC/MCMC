# Frente 5, medio paso 2: Sculptor contra el potencial débil (5A/5B/5D)

El contraste declarado de antemano (5E): los MISMOS parámetros (α₀⁻¹, ρ_c) — que solo entran por A = α₀⁻¹/ρ_c^(3/2) — deben explicar sistemas rotacionales y de presión; el fallo cuenta como falsación del mecanismo galáctico propuesto, no como llamada a ajustar por sistema.

**Datos** (procedencia y estatuto en `dynamics/dsph_data.py`): Walker et al. 2009 (AJ 137, 3100; ApJ 704, 1274); valores globales verificados vía resúmenes de búsqueda el 10-ago-2026; tabla original y perfil binado σ_los(R): ingesta pendiente (archivos bloqueados por el proxy de la sesión). σ_obs = 9.2 km/s (banda 9-10), R_half = 260 ± 39 pc, L_V ≈ 2.6e+06 L⊙ (derivada de M_V ≈ -11.2), Υ⋆ ∈ [1, 3] declarado.

## 1. Validación del solucionador

Jeans + proyección clavan la identidad analítica del Plummer isótropo con error relativo máximo 1.2e-03 (test permanente en `tests/test_jeans_dsph.py`).

## 2-4. El resultado central (β = 0)

| Υ⋆ | σ_N (solo estrellas) | σ_max dentro de (11.5) | techo puntual √2·σ_N | ρ_c exigido (α₀⁻¹=1e-6) [M⊙/pc³] | violación de (11.5) |
|---|---|---|---|---|---|
| 1 | 2.07 | 2.72 | 2.93 | 6.63e-01 | ×26 |
| 2 | 2.93 | 3.85 | 4.14 | 1.38e+00 | ×12 |
| 3 | 3.58 | 4.71 | 5.07 | 2.14e+00 | ×8 |

**Lectura.** (i) Solo estrellas: σ_N = 2.07-3.58 km/s según Υ⋆ — el déficit clásico frente a σ_obs = 9.2. (ii) El término débil de Cronos −c²∇ε_c NO puede cerrarlo: su propia cota (ec. 11.5, c²ε_c ≲ |Φ_N|) limita la subida a √2·σ_N ≤ 5.07 km/s incluso saturada puntualmente — a 3.9 km/s del borde inferior de la banda observada. (iii) La amplitud que Sculptor exige viola la cota por un factor ×8-×26 (según Υ⋆), y con α₀⁻¹ en su cota implica ρ_c = 0.7-2.1 M⊙/pc³ — una densidad de escala estelar, no cosmológica: como ε_c ∝ ρ^(3/2), cualquier sistema con densidades muy por debajo de ese ρ_c queda con ε_c despreciable, en tensión directa con el paso 5C si el mismo término tuviera que actuar en discos.

## 5. El objetivo que queda para ρ_id

Masa dinámica global (estimador estándar M_1/2 ≈ 4σ²R_half/G, Wolf et al. 2010; aproximación declarada, insensible a β al primer orden): **2.0e+07 M⊙** frente a M⋆(Υ⋆=2) = 5.2e+06 M⊙ — cociente ≈ 4. En la arquitectura del tratado, esa carga explicativa corresponde al sector ρ_id (el perfil cored que el Apéndice A asigna a las curvas de rotación), cuyo perfil a escala dSph NO está derivado: hueco declarado del frente.

## 6. Sensibilidades (Υ⋆ = 2)

| variación | σ_N | σ_max (11.5) | ρ_c exigido | violación |
|---|---|---|---|---|
| σ_obs = 9 | 2.93 | 3.85 | 1.42e+00 | ×12 |
| σ_obs = 9.2 | 2.93 | 3.85 | 1.38e+00 | ×12 |
| σ_obs = 10 | 2.93 | 3.85 | 1.22e+00 | ×15 |
| β = -0.3 | 2.93 | 3.85 | 1.38e+00 | ×12 |
| β = 0.3 | 2.93 | 3.85 | 1.38e+00 | ×12 |
| R_half = 221 | 3.17 | 4.18 | 2.27e+00 | ×10 |
| R_half = 299 | 2.73 | 3.59 | 8.95e-01 | ×14 |

Las filas β = ±0.3 son idénticas POR TEOREMA, no por descuido: el promedio total pesado por luminosidad ⟨σ_los²⟩ es independiente de la anisotropía en un sistema esférico (teorema virial proyectado; verificado como test permanente). El efecto de β vive en el PERFIL σ_los(R) — inaccesible hasta la ingesta de los datos binados, pendiente declarado. El veredicto (ii) es robusto en todo el barrido: la mayor σ_max dentro de la cota queda por debajo de la banda observada en todos los casos.

## Estatuto

condicional y de medio paso (frente 5): (a) el veredicto es EXACTO dentro del montaje declarado — trazador Plummer isótropo/β constante, datos globales (un número), fuente bariónica sola; el perfil binado σ_los(R), poblaciones múltiples y la ingesta de la tabla original quedan pendientes (procedencia en dynamics/dsph_data.py); (b) lo que queda falsado en este montaje es que el TÉRMINO DÉBIL DE CRONOS explique los dSph dentro de su cota (11.5) — no el modelo completo: el tratado asigna la fenomenología galáctica a ρ_id, que este medio paso convierte en objetivo cuantitativo (M_1/2 ≈ 2.0e+07 M⊙ dentro de ~260 pc); (c) el ρ_c aquí exigido (escala estelar) discrepa en órdenes de magnitud de la receta que la malla PM del programa necesitó (ρ_c ≈ umbral de colapso ~200× la media — nota computacional I, §3.5): la incógnita compartida del contraste 5E ya tiene dos medidas discrepantes; (d) el paso 5C (SPARC, sistemas rotacionales con la MISMA A) decidirá la falsación cruzada 5E.
