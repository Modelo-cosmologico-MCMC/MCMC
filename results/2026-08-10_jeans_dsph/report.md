# Frente 5, medio paso 2: Sculptor contra el potencial débil (5A/5B/5D)

El contraste declarado de antemano (5E): los MISMOS parámetros (α₀⁻¹, ρ_c) — que solo entran por A = α₀⁻¹/ρ_c^(3/2) — deben explicar sistemas rotacionales y de presión; el fallo cuenta como falsación del mecanismo galáctico propuesto, no como llamada a ajustar por sistema.

**Datos** (procedencia y estatuto en `dynamics/dsph_data.py`): Walker et al. 2009 (AJ 137, 3100; ApJ 704, 1274); valores globales verificados vía resúmenes de búsqueda el 10-ago-2026; tabla original y perfil binado σ_los(R): ingesta pendiente (archivos bloqueados por el proxy de la sesión). σ_obs = 9.2 km/s (banda 9-10), R_half = 260 ± 39 pc, L_V ≈ 2.6e+06 L⊙ (derivada de M_V ≈ -11.2), Υ⋆ ∈ [1, 3] declarado.

## 1. Validación del solucionador

Jeans + proyección clavan la identidad analítica del Plummer isótropo con error relativo máximo 1.2e-03 (test permanente en `tests/test_jeans_dsph.py`).

## 2-4. El resultado central (β = 0)

Las DOS LECTURAS de la ec. (11.5) — condición expuesta, no resuelta (docstring de `dynamics/weak_field.py`): (L) la desigualdad LITERAL acota solo α₀⁻¹ ≲ 1e-6, con ρ_c libre; (P) la condición de SUBDOMINANCIA de la que el tratado la deriva («para que no domine sobre la gravedad en halos, c²ε_c ≲ |Φ_N|», Cor. 11.3c), leída punto a punto en el sistema. No son equivalentes con ρ_c libre; el veredicto se publica bajo cada una.

| Υ⋆ | σ_N (solo estrellas) | σ_max subdominante (P) | techo puntual √2·σ_N | ρ_c máximo compatible (α₀⁻¹ ≤ 1e-6) [M⊙/pc³] | dominancia exigida c²ε_c/|Φ_N| |
|---|---|---|---|---|---|
| 1 | 2.07 | 2.72 | 2.93 | 6.63e-01 | ×26 |
| 2 | 2.93 | 3.85 | 4.14 | 1.38e+00 | ×12 |
| 3 | 3.58 | 4.71 | 5.07 | 2.14e+00 | ×8 |

**Lectura.** (i) Solo estrellas: σ_N = 2.07-3.58 km/s según Υ⋆ — el déficit clásico frente a σ_obs = 9.2. (ii) BAJO LA LECTURA P, el término débil de Cronos NO puede cerrarlo: la subdominancia limita la subida a √2·σ_N ≤ 5.07 km/s incluso saturada puntualmente — a 3.9 km/s del borde inferior de la banda observada. (iii) BAJO LA LECTURA L no hay violación numérica de la desigualdad — la amplitud exigida A_req es realizable con α₀⁻¹ ≪ 1e-6 y ρ_c pequeño, y ε_c se mantiene ≪ 1 —, pero entonces el término debe ser el potencial DOMINANTE del sistema: c²ε_c = ×8-×26·|Φ_N| (según Υ⋆), exactamente el régimen que la justificación declarada de la (11.5) excluye. El hecho invariante entre lecturas: EL TÉRMINO SOLO EXPLICA SCULPTOR DEJANDO DE SER UNA CORRECCIÓN SUBDOMINANTE DE CAMPO DÉBIL. (iv) Con α₀⁻¹ en su cota, la exigencia se traduce en la cota superior unilateral ρ_c ≤ 0.7-2.1 M⊙/pc³ (columna 5; para α₀⁻¹ menor, menor): Sculptor NO mide ρ_c — mide la amplitud combinada A = α₀⁻¹/ρ_c^(3/2), la incógnita compartida que el contraste 5E confrontará con SPARC.

## 5. El objetivo que queda para ρ_id

Masa dinámica global (estimador estándar M_1/2 ≈ 4σ²R_e/G = 3σ²r_1/2/G, Wolf et al. 2010; aproximación declarada, insensible a β al primer orden): **2.0e+07 M⊙** dentro del radio de media luz 3D r_1/2 ≈ (4/3)·R_e ≈ 347 pc (Plummer exacto: 1.305·a ≈ 339 pc — NO dentro del R_half proyectado de 260 pc), frente a M⋆(Υ⋆=2) = 5.2e+06 M⊙ — cociente ≈ 4. En la arquitectura del tratado, esa carga explicativa corresponde al sector ρ_id (el perfil cored que el Apéndice A asigna a las curvas de rotación), cuyo perfil a escala dSph NO está derivado: hueco declarado del frente.

## 6. Sensibilidades (Υ⋆ = 2)

| variación | σ_N | σ_max subdominante (P) | ρ_c máx. compatible (α₀⁻¹ ≤ 1e-6) | dominancia exigida |
|---|---|---|---|---|
| σ_obs = 9 | 2.93 | 3.85 | 1.42e+00 | ×12 |
| σ_obs = 9.2 | 2.93 | 3.85 | 1.38e+00 | ×12 |
| σ_obs = 10 | 2.93 | 3.85 | 1.22e+00 | ×15 |
| β = -0.3 | 2.93 | 3.85 | 1.38e+00 | ×12 |
| β = 0.3 | 2.93 | 3.85 | 1.38e+00 | ×12 |
| R_half = 221 | 3.17 | 4.18 | 2.27e+00 | ×10 |
| R_half = 299 | 2.73 | 3.59 | 8.95e-01 | ×14 |

Las filas β = ±0.3 son idénticas POR TEOREMA, no por descuido: el promedio total pesado por luminosidad ⟨σ_los²⟩ es independiente de la anisotropía en un sistema esférico (teorema virial proyectado; verificado como test permanente). El efecto de β vive en el PERFIL σ_los(R) — inaccesible hasta la ingesta de los datos binados, pendiente declarado. El veredicto (ii)/(iii) es robusto en todo el barrido: la mayor σ_max subdominante queda por debajo de la banda observada, y la dominancia exigida es ≥ ×8, en todos los casos.

## Estatuto

condicional y de medio paso (frente 5): (a) el veredicto es EXACTO dentro del montaje declarado — trazador Plummer isótropo/β constante, datos globales (un número), fuente bariónica sola; el perfil binado σ_los(R), poblaciones múltiples y la ingesta de la tabla original quedan pendientes (procedencia en dynamics/dsph_data.py); (b) lo medido, bajo cada lectura declarada de la (11.5): bajo P (subdominancia punto a punto) el término débil de Cronos queda falsado como explicación de Sculptor en este montaje; bajo L (desigualdad literal sobre α₀⁻¹, ρ_c libre) no hay violación numérica, pero el término solo alcanza σ_obs siendo el potencial dominante (×8-×26·|Φ_N|) — el régimen que la justificación declarada de la cota excluye. En ningún caso queda falsado el modelo completo: el tratado asigna la fenomenología galáctica a ρ_id, que este medio paso convierte en objetivo cuantitativo (M_1/2 ≈ 2.0e+07 M⊙ dentro de r_1/2 ≈ 347 pc); (c) Sculptor NO mide ρ_c — mide la amplitud A = α₀⁻¹/ρ_c^(3/2); con α₀⁻¹ en su cota eso da la cota superior unilateral ρ_c ≲ 1-2 M⊙/pc³, COMPATIBLE con la receta de validez de la malla PM del §3.5 (ρ_c ≈ umbral de colapso, en unidades de código y sin anclaje físico en el repositorio — es un requisito de régimen, no una medida): el contraste cruzado 5E sigue plenamente abierto; (d) el paso 5C (SPARC, sistemas rotacionales con la MISMA A) lo decidirá.
